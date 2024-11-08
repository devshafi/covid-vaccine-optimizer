# optimization.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class OptimizationParameters:
    alpha: float  # Base reward for vaccination
    beta: float   # Priority weight
    gamma: float  # Distance penalty

class VaccineOptimizer:
    def __init__(
        self,
        persons: pd.DataFrame,
        centers: pd.DataFrame,
        distances: np.ndarray,
        n_vaccines: int,
        params: OptimizationParameters
    ):
        self.persons = persons
        self.centers = centers
        self.distances = distances
        self.n_vaccines = n_vaccines
        self.params = params
        
    def create_base_model(self) -> gp.Model:
        """Creates base model with common constraints"""
        model = gp.Model("Vaccine_Distribution")
        
        # Sets
        I = range(len(self.centers))    # Distribution centers
        J = range(sum(self.centers['staff'])) # Healthcare workers
        K = range(len(self.persons))    # Persons
        
        # Decision Variables
        x = model.addVars(I, J, K, vtype=GRB.BINARY, name="x")
        
        # Constraints
        # Each worker can vaccinate at most one person
        for i in I:
            for j in J:
                model.addConstr(gp.quicksum(x[i,j,k] for k in K) <= 1)
                
        # Each person gets at most one vaccine
        for k in K:
            model.addConstr(gp.quicksum(x[i,j,k] for i in I for j in J) <= 1)
            
        # Cannot exceed vaccine supply
        model.addConstr(gp.quicksum(x[i,j,k] for i in I for j in J for k in K) <= self.n_vaccines)
        
        return model, x

    def solve_bvdm(self) -> Dict:
        """Basic Vaccine Distribution Model"""
        model, x = self.create_base_model()
        
        # Objective: Maximize number of vaccinations
        obj = gp.quicksum(self.params.alpha * x[i,j,k] 
                         for i in range(len(self.centers))
                         for j in range(sum(self.centers['staff']))
                         for k in range(len(self.persons)))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        model.optimize()
        return self._get_solution(model, x)
    
    def solve_pvdm(self) -> Dict:
        """Priority-based Vaccine Distribution Model"""
        model, x = self.create_base_model()
        
        # Objective: Maximize vaccinations considering priority
        obj = gp.quicksum((self.params.alpha + self.params.beta * self.persons.loc[k, 'priority']) * x[i,j,k]
                            for i in range(len(self.centers))
                            for j in range(sum(self.centers['staff']))
                            for k in range(len(self.persons)))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        model.optimize()
        return self._get_solution(model, x)

    def solve_dvdm(self) -> Dict:
        """Distance-based Vaccine Distribution Model"""
        model, x = self.create_base_model()
        
        # Objective: Maximize vaccinations considering distance
        obj = gp.quicksum((self.params.alpha - self.params.gamma * self.distances[k,i]) * x[i,j,k]
                            for i in range(len(self.centers))
                            for j in range(sum(self.centers['staff']))
                            for k in range(len(self.persons)))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        model.optimize()
        return self._get_solution(model, x)

    def solve_pdvdm(self) -> Dict:
        """Priority and Distance-based Vaccine Distribution Model"""
        model, x = self.create_base_model()
        
        # Objective: Maximize considering both priority and distance
        obj = gp.quicksum((self.params.alpha + 
                            self.params.beta * self.persons.loc[k, 'priority'] - 
                            self.params.gamma * self.distances[k,i]) * x[i,j,k]
                            for i in range(len(self.centers))
                            for j in range(sum(self.centers['staff']))
                            for k in range(len(self.persons)))
        model.setObjective(obj, GRB.MAXIMIZE)
        
        model.optimize()
        return self._get_solution(model, x)

    def _get_solution(self, model: gp.Model, x: gp.tupledict) -> Dict:
        """Extract solution details"""
        if model.Status == GRB.OPTIMAL:
            solution = {
                'objective_value': model.ObjVal,
                'assignments': [],
                'total_vaccinated': 0
            }
            
            for i in range(len(self.centers)):
                for j in range(sum(self.centers['staff'])):
                    for k in range(len(self.persons)):
                        if x[i,j,k].X > 0.5:  # Binary variable threshold
                            solution['assignments'].append({
                                'center': i,
                                'worker': j,
                                'person': k,
                                'priority': self.persons.loc[k, 'priority'],
                                'distance': self.distances[k,i]
                            })
                            solution['total_vaccinated'] += 1
                            
            return solution
        return {'error': 'No optimal solution found'}