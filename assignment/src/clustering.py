# clustering.py
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from typing import Tuple

class ClusteringService:
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def calculate_distances(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def get_distance_matrix(self, persons: pd.DataFrame, hospitals: pd.DataFrame) -> np.ndarray:
        """Create distance matrix between persons and hospitals"""
        n_persons = len(persons)
        n_hospitals = len(hospitals)
        distances = np.zeros((n_persons, n_hospitals))
        
        for i in range(n_persons):
            person_loc = persons[['x', 'y']].iloc[i].values
            for j in range(n_hospitals):
                hospital_loc = hospitals[['x', 'y']].iloc[j].values
                distances[i,j] = self.calculate_distances(person_loc, hospital_loc)
                
        return distances
    
    def select_distribution_centers(self, hospitals: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Select distribution centers using K-medoids clustering"""
        locations = hospitals[['x', 'y']].values
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        clusters = kmedoids.fit(locations)
        selected_centers = hospitals.iloc[kmedoids.medoid_indices_]
        
        return selected_centers, clusters.labels_