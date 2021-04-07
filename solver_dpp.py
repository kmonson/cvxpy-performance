# Most of this code was borrowed from StorageVet and adapted for our purposes
# https://github.com/epri-dev/StorageVET/blob/master/Scenario.py

from __future__ import annotations

import datetime
import os
import time
from typing import Any, Dict

import cvxpy
import numpy as np
import pandas as pd


CVXPY_VERBOSE = bool(os.environ.get('CVXPY_VERBOSE'))

CVXPY_SOLVER = cvxpy.ECOS

cvxpy.enable_warnings()


# def rep_cvxpy(obj):
#     if isinstance(obj, cvxpy.expressions.expression.Expression):
#         return obj.name(), obj, rep_cvxpy(obj.value)
#
#     if isinstance(obj, cvxpy.constraints.constraint.Constraint):
#         return obj.name(), obj, rep_cvxpy(obj.args)
#
#     if isinstance(obj, cvxpy.problems.objective.Objective):
#         return obj.NAME, obj, rep_cvxpy(obj.args)
#
#     if isinstance(obj, dict):
#         return {k: rep_cvxpy(v) for k, v in obj.items()}
#
#     if isinstance(obj, (list, tuple)):
#         return [rep_cvxpy(v) for v in obj]
#
#     return obj


class DAEnergyTimeShift:
    def __init__(self, price: pd.DataFrame, dt: float) -> None:
        self.dt = dt
        self.price = price

    def objective_function(self, variables: Dict[str, Any],
                           generation: cvxpy.Variable, annuity_scalar: float = 1.0) -> Dict[str, Any]:

        p_da = cvxpy.Parameter(self.price.index.size, name='da price', value=[i[0] for i in self.price.values])
        return {'DA ETS': cvxpy.sum(-p_da @ variables['dis'] + p_da @ variables['ch'] -
                                    p_da @ generation) * annuity_scalar * self.dt}


class PVGen:
    def __init__(self, pv_prod: pd.Series, pv_capacity: float) -> None:
        self.generation = pv_prod
        self.inv_max = pv_capacity

    def objective_constraints(self, variables: Dict[str, Any], mask: pd.Series) -> list:
        # For this test mask is all true and no filtering takes place
        constraints = [
            variables['pv_out'] - self.generation[mask] <= 0,
            variables['ch'] - variables['pv_out'] <= 0,
            variables['pv_out'] - self.inv_max <= 0,
            -self.inv_max - variables['pv_out'] <= 0,
        ]
        return constraints


class BESS:
    def __init__(self, power_capacity: float, energy_capacity: float,
                 rte: float, daily_cycle_limit: float, dt: float, soc_target: float) -> None:
        self.ene_max_rated = energy_capacity
        self.dis_max_rated = power_capacity
        self.ch_max_rated = power_capacity
        self.ulsoc = 1
        self.llsoc = 0
        self.rte = rte
        self.daily_cycle_limit = daily_cycle_limit
        self.dt = dt  # granularity of simulation
        self.soc_target = soc_target

    def objective_constraints(self, variables: Dict[str, Any],
                              mask: pd.Series, reservations: Dict[str, Any]) -> list:
        ene_target = self.soc_target * self.ulsoc * self.ene_max_rated

        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']

        # create cvx parameters of control constraints (this improves readability in cvx costs and better handling)
        # For this test mask is all true and no filtering takes place
        # also size is always 8760
        size = int(np.sum(mask))
        ene_max = self.ulsoc * self.ene_max_rated
        ene_min = self.llsoc * self.ene_max_rated
        ch_max = self.ch_max_rated
        ch_min = 0.0
        dis_max = self.dis_max_rated
        dis_min = 0.0

        # energy at the end of the last time step (makes sure that the end of the last time step is ENE_TARGET
        e_res = reservations['E']
        constraints = [
            (ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - e_res == 0,

            # energy generally for every time step
            ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - e_res == 0,

            # energy at the beginning of the optimization window -- handles rolling window
            ene[0] - ene_target == 0,

            # Keep energy in bounds determined in the constraints configuration function
            # making sure our storage meets control constraints
            ene[:-1] - ene_max <= 0,
            ene_min - ene[1:] <= 0,

            # Keep charge and discharge power levels within bounds
            -ch_max + ch - dis <= 0,
            -ch + dis <= 0,

            ch - cvxpy.multiply(ch_max, on_c) <= 0,
            dis - cvxpy.multiply(dis_max, on_d) <= 0,

            # removing the band in between ch_min and dis_min that the battery will not operate in
            cvxpy.multiply(ch_min, on_c) - ch + reservations['C_min'] <= 0,
            cvxpy.multiply(dis_min, on_d) - dis + reservations['D_min'] <= 0,
        ]

        # For this test mask is all true and no filtering takes place
        days = mask.loc[mask].index.dayofyear
        constraints.extend(
            cvxpy.sum(dis[day_mask] * self.dt + e_res) -
            self.ene_max_rated * self.daily_cycle_limit <= 0
            for day_mask in (day == days for day in days.unique()))
        return constraints


class StorageSolver:
    def __init__(self, curve: pd.DataFrame) -> None:
        self.curve = curve
        self.daily_cycle_limit = 2.0
        self.dt = 1.0
        self.soc_target = 0.0
        self.rte = 0.87

    def optimization_problem(self, year: int, pv_total_plant: pd.DataFrame, pv_ac_plant: pd.Series,
                             power_capacity: float, energy_capacity: float) -> pd.DataFrame:
        # For this test mask is all True
        mask: pd.Series = pv_total_plant > -200
        # For this test size is always 8760
        size = int(np.sum(mask))

        ##########################################################################
        # COLLECT OPTIMIZATION VARIABLES & POWER/ENERGY RESERVATIONS/THROUGHPUTS #
        ##########################################################################

        # Add optimization variables for each technology
        generation = cvxpy.Variable(shape=size, name='pv_out', nonneg=True)
        variables = {
            'ene': cvxpy.Variable(shape=size, name='ene'),  # Energy at the end of the time step
            'dis': cvxpy.Variable(shape=size, name='dis'),  # Discharge Power, kW during the previous time step
            'ch': cvxpy.Variable(shape=size, name='ch'),  # Charge Power, kW during the previous time step
            'pv_out': generation,
            'on_c': 1,
            'on_d': 1,
        }

        # Calculate system generation
        reservations = {
            'C_max': 0,  # default power and energy reservations
            'C_min': 0,
            'D_max': 0,
            'D_min': 0,
            'E': 0
        }

        #################################################
        # COLLECT OPTIMIZATION CONSTRAINTS & OBJECTIVES #
        #################################################

        da = DAEnergyTimeShift(self.curve[size * year:size * (year + 1)], self.dt)
        expression = da.objective_function(variables, generation)

        pv = PVGen(pv_ac_plant, round(max(pv_ac_plant), 0))
        bess = BESS(power_capacity, energy_capacity, self.rte,
                    self.daily_cycle_limit, self.dt, self.soc_target)
        constraints = [
            -variables['dis'] + variables['ch'] - generation <= 0,
            *pv.objective_constraints(variables, mask),
            *bess.objective_constraints(variables, mask, reservations),
        ]

        objective = cvxpy.Minimize(sum(expression.values()))

        print("Constructing problem.")
        prob = cvxpy.Problem(objective, constraints)

        print("Solving problem")
        start_time = time.perf_counter()
        prob.solve(solver=CVXPY_SOLVER, verbose=True)
        end_time = time.perf_counter()
        print(f"Total time solving problem {end_time - start_time}")

        result_str = f'Optimization problem was {prob.status}'
        print(result_str)


if __name__ == "__main__":
    data = pd.read_csv('test_data.csv', index_col=0)
    curve = pd.read_csv("curve.csv")
    data = data['Values']
    data.index = pd.to_datetime(data.index)
    for window in (1, 2, 3, 7, 14, 30, 365):
        print(f"{20*'*'}  {window}  {20*'*'}")
        w = window * 24
        c = curve[:w]
        d = data[:w]
        solver = StorageSolver(c)
        solver.optimization_problem(0, d, d, 100, 400)



