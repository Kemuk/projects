from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from house import House


class OptimizationComparison:
    """
    Utility class for comparing multiple rent optimization methods.
    
    Handles running optimizers, combining results, calculating metrics,
    and exporting to CSV files.
    """
    
    def __init__(self, house: House, output_dir: str = "results"):
        """
        Initialize the comparison utility.
        
        Args:
            house: House object containing room data
            output_dir: Directory to save output files
        """
        self.house = house
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.optimizers = []
        self.results_df = None
        self.fairness_df = None
        self.summary_df = None
    
    def add_optimizer(self, optimizer):
        """Add an optimizer to the comparison."""
        self.optimizers.append(optimizer)
    
    def add_optimizers(self, optimizers: List):
        """Add multiple optimizers to the comparison."""
        self.optimizers.extend(optimizers)
    
    def run_comparison(self) -> Dict[str, pd.DataFrame]:
        """
        Run all optimizers and generate comparison results.
        
        Returns:
            Dictionary containing different comparison DataFrames
        """
        if not self.optimizers:
            raise ValueError("No optimizers added to comparison")
        
        # Run all optimizations and collect results
        all_results = []
        all_fairness = []
        
        for optimizer in self.optimizers:
            # Get detailed results
            result_df = optimizer.get_result_dataframe()
            all_results.append(result_df)
            
            # Get fairness metrics
            fairness_metrics = optimizer.calculate_fairness_metrics()
            all_fairness.append(fairness_metrics)
        
        # Combine all results
        self.results_df = pd.concat(all_results, ignore_index=True)
        self.fairness_df = pd.DataFrame(all_fairness)
        
        # Create summary comparison
        self._create_summary_comparison()
        
        return {
            'detailed_results': self.results_df,
            'fairness_metrics': self.fairness_df,
            'summary_comparison': self.summary_df
        }
    
    def _create_summary_comparison(self):
        """Create a summary comparison table."""
        summary_data = []
        
        methods = self.results_df['Method'].unique()
        rooms = self.results_df['Room_ID'].unique()
        
        # Create pivot table for easy comparison
        pivot_costs = self.results_df.pivot(index='Room_ID', 
                                           columns='Method', 
                                           values='Total_Monthly_Cost')
        
        pivot_individual = self.results_df.pivot(index='Room_ID', 
                                                columns='Method', 
                                                values='Individual_Cost')
        
        # Add room information
        room_info = self.results_df[['Room_ID', 'Size_sqm', 'Desirability_Score']].drop_duplicates()
        room_info = room_info.set_index('Room_ID')
        
        # Combine into summary
        self.summary_df = pd.concat([room_info, pivot_costs], axis=1)
        self.summary_df.reset_index(inplace=True)
    
    def export_to_csv(self, filename_prefix: str = "rent_price_methods"):
        """
        Export comparison results to CSV files.
        
        Args:
            filename_prefix: Prefix for output filenames
        """
        if self.results_df is None:
            self.run_comparison()
        
        # Export detailed results
        detailed_file = self.output_dir / f"{filename_prefix}_detailed.csv"
        self.results_df.to_csv(detailed_file, index=False)
        print(f"📊 Detailed results exported to: {detailed_file}")
        
        # Export fairness metrics
        fairness_file = self.output_dir / f"{filename_prefix}_fairness.csv"
        self.fairness_df.to_csv(fairness_file, index=False)
        print(f"📈 Fairness metrics exported to: {fairness_file}")
        
        # Export summary comparison
        summary_file = self.output_dir / f"{filename_prefix}_summary.csv"
        self.summary_df.to_csv(summary_file, index=False)
        print(f"📋 Summary comparison exported to: {summary_file}")
        
        return {
            'detailed': detailed_file,
            'fairness': fairness_file, 
            'summary': summary_file
        }
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary."""
        if self.results_df is None:
            self.run_comparison()
        
        print("\n🏠 RENT OPTIMIZATION COMPARISON")
        print("=" * 80)
        
        # Quick comparison table
        print(f"\n💰 TOTAL MONTHLY COSTS BY METHOD")
        print("-" * 60)
        
        # Create a clean comparison table
        methods = self.results_df['Method'].unique()
        rooms = sorted(self.results_df['Room_ID'].unique())
        
        # Header
        header = f"{'Method':<25}"
        for room in rooms:
            header += f"{room:<10}"
        print(header)
        print("-" * len(header))
        
        # Method rows
        for method in methods:
            method_data = self.results_df[self.results_df['Method'] == method]
            row = f"{method:<25}"
            for room in rooms:
                cost = method_data[method_data['Room_ID'] == room]['Total_Monthly_Cost'].iloc[0]
                row += f"£{cost:<9.0f}"
            print(row)
        
        # Add desirability scores
        print(f"\n{'Desirability Scores':<25}", end="")
        for room in rooms:
            score = self.results_df[self.results_df['Room_ID'] == room]['Desirability_Score'].iloc[0]
            print(f"{score:<10.1f}", end="")
        print()
        
        # Add room sizes
        print(f"{'Room Sizes (sqm)':<25}", end="")
        for room in rooms:
            size = self.results_df[self.results_df['Room_ID'] == room]['Size_sqm'].iloc[0]
            print(f"{size:<10.1f}", end="")
        print()
        
        # Fairness metrics summary
        print(f"\n📊 FAIRNESS METRICS COMPARISON")
        print("-" * 60)
        fairness_summary = self.fairness_df[['Method', 'Range_Max_Min', 'Standard_Deviation', 
                                           'Gini_Coefficient', 'Max_Deviation_from_Mean']]
        print(fairness_summary.to_string(index=False, float_format='%.2f'))
    
    def create_visualization(self, save_plots: bool = True):
        """
        Create visualization plots for the comparison.
        
        Args:
            save_plots: Whether to save plots to files
        """
        if self.results_df is None:
            self.run_comparison()
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rent Optimization Methods Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Total costs by method
        pivot_costs = self.results_df.pivot(index='Room_ID', columns='Method', values='Total_Monthly_Cost')
        pivot_costs.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Total Monthly Costs by Method')
        ax1.set_ylabel('Monthly Rent (£)')
        ax1.axhline(y=self.house.target_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Target Mean (£{self.house.target_mean:.0f})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviation from target mean
        pivot_deviation = self.results_df.pivot(index='Room_ID', columns='Method', values='Deviation_from_Mean')
        pivot_deviation.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Deviation from Target Mean')
        ax2.set_ylabel('Deviation (£)')
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fairness metrics comparison
        fairness_metrics = ['Standard_Deviation', 'Range_Max_Min', 'Gini_Coefficient']
        fairness_subset = self.fairness_df[['Method'] + fairness_metrics].set_index('Method')
        fairness_subset.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Fairness Metrics Comparison')
        ax3.set_ylabel('Metric Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cost per square meter
        pivot_cost_per_sqm = self.results_df.pivot(index='Room_ID', columns='Method', values='Cost_per_sqm')
        pivot_cost_per_sqm.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Cost per Square Meter')
        ax4.set_ylabel('£ per sqm')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.output_dir / "rent_optimization_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
    
    def get_best_method_by_metric(self, metric: str = 'Gini_Coefficient') -> str:
        """
        Get the best method according to a fairness metric.
        
        Args:
            metric: Fairness metric to use for comparison
            
        Returns:
            Name of the best method
        """
        if self.fairness_df is None:
            self.run_comparison()
        
        # Lower is better for most metrics
        best_idx = self.fairness_df[metric].idxmin()
        return self.fairness_df.loc[best_idx, 'Method']
    
    def get_recommendations(self) -> Dict[str, str]:
        """
        Get method recommendations based on different criteria.
        
        Returns:
            Dictionary of recommendations for different use cases
        """
        if self.fairness_df is None:
            self.run_comparison()
        
        recommendations = {
            'Most Equal (lowest Gini)': self.get_best_method_by_metric('Gini_Coefficient'),
            'Lowest Range': self.get_best_method_by_metric('Range_Max_Min'),
            'Most Stable (lowest std dev)': self.get_best_method_by_metric('Standard_Deviation'),
            'Closest to Proportional': self.get_best_method_by_metric('Avg_Deviation_from_Proportional'),
            'Closest to Equal Mean': self.get_best_method_by_metric('Avg_Deviation_from_Mean')
        }
        
        return recommendations
    
    def handle_single_optimizer(self, optimizer) -> pd.DataFrame:
        """
        Handle the base case of only one optimizer.
        
        Args:
            optimizer: Single optimizer to run
            
        Returns:
            Results DataFrame
        """
        self.add_optimizer(optimizer)
        results = self.run_comparison()
        
        # Export with base case naming
        self.export_to_csv("rent_price_methods")
        
        return results['detailed_results']