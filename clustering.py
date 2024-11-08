# clustering.py
from sklearn_extra.cluster import KMedoids
import numpy as np

def perform_kmedoids(hospitals, n_clusters):
    """
    Perform K-medoids clustering to select distribution centers
    """
    locations = hospitals[['x', 'y']].values
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    clusters = kmedoids.fit(locations)
    
    # Get selected distribution centers
    selected_centers = hospitals.iloc[kmedoids.medoid_indices_]
    return selected_centers