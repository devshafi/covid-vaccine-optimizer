# main.py
import numpy as np
from src.data_generator import DataGenerator
from src.clustering import ClusteringService
from src.optimization import VaccineOptimizer, OptimizationParameters
from src.visualization import VaccinationVisualizer
import matplotlib.pyplot as plt

def run_scenario(n_persons: int, n_hospitals: int, n_vaccines: int, n_centers: int):
    """Run complete vaccine distribution scenario"""
    
    # Generate synthetic data
    data_gen = DataGenerator(n_persons, n_hospitals, n_vaccines)
    persons, hospitals = data_gen.generate()
    
    # Perform clustering
    clustering = ClusteringService(n_clusters=n_centers)
    centers, labels = clustering.select_distribution_centers(hospitals)
    distances = clustering.get_distance_matrix(persons, centers)
    
    # Set optimization parameters
    params = OptimizationParameters(
        alpha=0.25 * n_persons,
        beta=0.25 * n_persons,
        gamma=1.0
    )
    
    # Create optimizer
    optimizer = VaccineOptimizer(persons, centers, distances, n_vaccines, params)
    
    # Run all models
    results = {
        'B-VDM': optimizer.solve_bvdm(),
        'P-VDM': optimizer.solve_pvdm(),
        'D-VDM': optimizer.solve_dvdm(),
        'PD-VDM': optimizer.solve_pdvdm()
    }
    
    # Visualize results
    visualizer = VaccinationVisualizer(results)
    
    # Generate and save plots
    priority_plot = visualizer.plot_priority_distribution()
    priority_plot.savefig(f'priority_dist_{n_persons}p_{n_hospitals}h_{n_vaccines}v.png')
    
    total_plot = visualizer.plot_total_vaccinations()
    total_plot.savefig(f'total_vac_{n_persons}p_{n_hospitals}h_{n_vaccines}v.png')
    
    distance_plot = visualizer.plot_distance_distribution()
    distance_plot.savefig(f'distance_dist_{n_persons}p_{n_hospitals}h_{n_vaccines}v.png')
    
    return results

if __name__ == "__main__":
    # Test scenarios
    scenarios = [
        (100, 10, 80, 5),   # 100 persons, 10 hospitals, 80 vaccines, 5 centers
        (200, 15, 150, 8),  # 200 persons, 15 hospitals, 150 vaccines, 8 centers
        (300, 20, 250, 10)  # 300 persons, 20 hospitals, 250 vaccines, 10 centers
    ]
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")
        results = run_scenario(*scenario)
        for model, result in results.items():
            print(f"{model}: {result['total_vaccinated']} vaccinated")