from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np
from house import House


class BaseOptimizer(ABC):
    """
    Abstract base class for all rent optimization algorithms.
    
    All optimizers take a House object and return a DataFrame with allocation results.
    """
    
    def __init__(self, house: House, name: str):
        """
        Initialize the optimizer.
        
        Args:
            house: House object containing room and cost information
            name: Name of the optimization method (e.g., "Proportional", "Quadratic")
        """
        self.house = house
        self.name = name
        self.allocation = {}
        self.result_df = None
    
    @abstractmethod
    def optimize(self) -> Dict[str, float]:
        """
        Perform the optimization and return the allocation.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        pass
    
    def get_result_dataframe(self) -> pd.DataFrame:
        """
        Get optimization results as a DataFrame.
        
        Returns:
            DataFrame with detailed allocation results
        """
        if not self.allocation:
            self.allocation = self.optimize()
        
        if self.result_df is not None:
            return self.result_df
        
        data = []
        for room in self.house.individual_rooms:
            room_id = room.room_id
            total_cost = self.allocation.get(room_id, 0)
            individual_cost = total_cost - self.house.shared_cost_per_person
            deviation_from_mean = total_cost - self.house.target_mean
            
            # Calculate deviation from proportional
            proportional_cost = room.total_cost  # This is the proportional cost
            deviation_from_proportional = total_cost - proportional_cost
            
            data.append({
                'Method': self.name,
                'Room_ID': room_id,
                'Size_sqm': room.size,
                'Desirability_Score': room.desirability_score or 0,
                'Individual_Cost': round(individual_cost, 2),
                'Shared_Cost_Portion': round(self.house.shared_cost_per_person, 2),
                'Total_Monthly_Cost': round(total_cost, 2),
                'Deviation_from_Mean': round(deviation_from_mean, 2),
                'Deviation_from_Proportional': round(deviation_from_proportional, 2),
                'Cost_per_sqm': round(total_cost / room.size, 2)
            })
        
        self.result_df = pd.DataFrame(data)
        return self.result_df
    
    def calculate_fairness_metrics(self) -> Dict[str, float]:
        """
        Calculate various fairness metrics for the allocation.
        
        Returns:
            Dictionary of fairness metrics
        """
        if not self.allocation:
            self.allocation = self.optimize()
        
        cost_values = list(self.allocation.values())
        proportional_costs = self.house.get_proportional_allocation()
        
        # Deviation from target mean
        mean_deviations = [abs(cost - self.house.target_mean) for cost in cost_values]
        avg_deviation_from_mean = np.mean(mean_deviations)
        max_deviation_from_mean = max(mean_deviations)
        
        # Deviation from proportional
        prop_deviations = [abs(self.allocation[room_id] - proportional_costs[room_id]) 
                          for room_id in self.allocation.keys()]
        avg_deviation_from_prop = np.mean(prop_deviations)
        max_deviation_from_prop = max(prop_deviations)
        
        # Standard statistical measures
        variance = np.var(cost_values)
        std_dev = np.std(cost_values)
        cost_range = max(cost_values) - min(cost_values)
        
        # Coefficient of variation (normalized measure of dispersion)
        cv = std_dev / np.mean(cost_values) if np.mean(cost_values) > 0 else 0
        
        # Gini coefficient (measure of inequality)
        gini = self._calculate_gini_coefficient(cost_values)
        
        return {
            'Method': self.name,
            'Avg_Deviation_from_Mean': round(avg_deviation_from_mean, 2),
            'Max_Deviation_from_Mean': round(max_deviation_from_mean, 2),
            'Avg_Deviation_from_Proportional': round(avg_deviation_from_prop, 2),
            'Max_Deviation_from_Proportional': round(max_deviation_from_prop, 2),
            'Standard_Deviation': round(std_dev, 2),
            'Range_Max_Min': round(cost_range, 2),
            'Variance': round(variance, 2),
            'Coefficient_of_Variation': round(cv, 4),
            'Gini_Coefficient': round(gini, 4),
            'Total_Rent_Check': round(sum(cost_values), 2)
        }
    
    def _calculate_gini_coefficient(self, values):
        """Calculate the Gini coefficient for measuring inequality."""
        sorted_values = sorted(values)
        n = len(values)
        cumulative_values = np.cumsum(sorted_values)
        return (n + 1 - 2 * sum(cumulative_values) / cumulative_values[-1]) / n
    
    def validate_allocation(self) -> bool:
        """
        Validate that the allocation sums to the total rent.
        
        Returns:
            True if allocation is valid
        """
        if not self.allocation:
            self.allocation = self.optimize()
        
        return self.house.validate_allocation(self.allocation)
    
    def print_summary(self):
        """Print a summary of the optimization results."""
        if not self.allocation:
            self.allocation = self.optimize()
        
        print(f"\n💰 {self.name.upper()} OPTIMIZATION RESULTS")
        print("=" * 60)
        
        for room in self.house.individual_rooms:
            room_id = room.room_id
            total_cost = self.allocation[room_id]
            individual_cost = total_cost - self.house.shared_cost_per_person
            deviation = total_cost - self.house.target_mean
            
            print(f"{room_id} ({room.size} sqm):")
            print(f"  Individual room: £{individual_cost:.2f}")
            print(f"  Shared portion:  £{self.house.shared_cost_per_person:.2f}")
            print(f"  TOTAL MONTHLY:   £{total_cost:.2f} ({deviation:+.0f} from £{self.house.target_mean:.0f} mean)")
            if room.desirability_score:
                print(f"  Desirability:    {room.desirability_score:.1f}/10")
            print()
        
        # Validation and metrics
        total_check = sum(self.allocation.values())
        print(f"✅ Total rent check: £{total_check:.2f} (target: £{self.house.total_rent})")
        
        metrics = self.calculate_fairness_metrics()
        print(f"📊 Range: £{metrics['Range_Max_Min']} | Std Dev: £{metrics['Standard_Deviation']:.2f} | Gini: {metrics['Gini_Coefficient']:.3f}")
    
    def apply_to_house(self):
        """Apply the optimization results to the house object."""
        if not self.allocation:
            self.allocation = self.optimize()
        
        self.house.apply_allocation(self.allocation)
    
    def get_costs_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of costs broken down by room.
        
        Returns:
            Dictionary with detailed cost breakdown
        """
        if not self.allocation:
            self.allocation = self.optimize()
        
        summary = {}
        for room in self.house.individual_rooms:
            room_id = room.room_id
            total_cost = self.allocation[room_id]
            individual_cost = total_cost - self.house.shared_cost_per_person
            
            summary[room_id] = {
                'size': room.size,
                'individual_cost': round(individual_cost, 2),
                'shared_cost': round(self.house.shared_cost_per_person, 2),
                'total_cost': round(total_cost, 2),
                'cost_per_sqm': round(total_cost / room.size, 2),
                'desirability': room.desirability_score or 0
            }
        
        return summary
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', house={self.house})"