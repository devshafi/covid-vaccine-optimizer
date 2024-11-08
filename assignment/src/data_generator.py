import numpy as np
import pandas as pd
from typing import Tuple

class DataGenerator:
    def __init__(self, n_persons: int, n_hospitals: int, n_vaccines: int, seed: int = 42):
        self.n_persons = n_persons
        self.n_hospitals = n_hospitals
        self.n_vaccines = n_vaccines
        np.random.seed(seed)
        
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Generate person data
        persons = pd.DataFrame({
            'id': range(self.n_persons),
            'x': np.random.uniform(0, 100, self.n_persons),
            'y': np.random.uniform(0, 100, self.n_persons),
            'priority': np.random.randint(1, 6, self.n_persons)  # Priority 1-5
        })
        
        # Generate hospital data
        hospitals = pd.DataFrame({
            'id': range(self.n_hospitals),
            'x': np.random.uniform(0, 100, self.n_hospitals),
            'y': np.random.uniform(0, 100, self.n_hospitals),
            'staff': np.random.randint(3, 8, self.n_hospitals)  # 3-7 staff per hospital
        })
        
        return persons, hospitals