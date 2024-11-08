import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import seaborn as sns

class VaccinationVisualizer:
    def __init__(self, results: Dict[str, Dict]):
        """
        Initialize visualizer with results from all models
        results: Dictionary with keys 'BVDM', 'PVDM', 'DVDM', 'PDVDM'
        """
        self.results = results
        self.models = ['B-VDM', 'P-VDM', 'D-VDM', 'PD-VDM']
        
    def plot_priority_distribution(self):
        """Plot number of vaccinated individuals by priority level"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get all unique priority levels across all models
        all_priorities = set()
        for model in self.models:
            if self.results[model]['assignments']:
                assignments = pd.DataFrame(self.results[model]['assignments'])
                all_priorities.update(assignments['priority'].unique())
        
        all_priorities = sorted(list(all_priorities))
        x = np.arange(len(all_priorities))
        width = 0.2
        
        for i, model in enumerate(self.models):
            if self.results[model]['assignments']:
                assignments = pd.DataFrame(self.results[model]['assignments'])
                counts = assignments['priority'].value_counts().reindex(all_priorities).fillna(0)
                ax.bar(x + i*width, counts.values, width, label=model)
        
        ax.set_xlabel('Priority Level')
        ax.set_ylabel('Number of Vaccinated Individuals')
        ax.set_title('Distribution of Vaccinations by Priority Level')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(all_priorities)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_total_vaccinations(self):
        """Plot total number of vaccinations for each model"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        totals = [self.results[model]['total_vaccinated'] for model in self.models]
        
        ax.bar(self.models, totals)
        ax.set_ylabel('Total Vaccinations')
        ax.set_title('Total Vaccinations by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_distance_distribution(self):
        """Plot distribution of travel distances"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model in self.models:
            distances = [a['distance'] for a in self.results[model]['assignments']]
            sns.kdeplot(data=distances, label=model)
            
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Travel Distances')
        ax.legend()
        plt.tight_layout()
        return fig