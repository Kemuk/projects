from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Data class to hold optimization results for easy serialization."""
    method_name: str
    allocation: Dict[str, float]
    fairness_metrics: Dict[str, float]
    parameters: Dict[str, Any]  # Store optimizer-specific parameters
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'method_name': self.method_name,
            'allocation': self.allocation,
            'fairness_metrics': self.fairness_metrics,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationResult':
        """Create from dictionary for JSON deserialization."""
        return cls(
            method_name=data['method_name'],
            allocation=data['allocation'],
            fairness_metrics=data['fairness_metrics'],
            parameters=data.get('parameters', {})
        )
    
    def get_display_data(self, house) -> Dict:
        """Get result data formatted for UI display."""
        costs = list(self.allocation.values())
        
        return {
            'method_name': self.method_name,
            'total_rent': sum(costs),
            'min_cost': min(costs),
            'max_cost': max(costs),
            'cost_range': max(costs) - min(costs),
            'average_cost': np.mean(costs),
            'std_deviation': self.fairness_metrics.get('Standard_Deviation', 0),
            'gini_coefficient': self.fairness_metrics.get('Gini_Coefficient', 0),
            'fairness_score': self.fairness_metrics.get('Fairness_Score', 0),
            'spread_percent': self.get_spread_percentage(),
            'room_allocations': self.allocation
        }
    
    def get_spread_percentage(self) -> Optional[float]:
        """Get spread percentage if available from parameters."""
        # Check if lambda parameter exists and convert to spread
        if 'lambda_param' in self.parameters:
            lambda_val = self.parameters['lambda_param']
            # Convert lambda (0-10) to spread percentage (0-100)
            return (lambda_val / 10.0) * 100.0
        return None
    
    def format_costs_for_display(self) -> Dict[str, str]:
        """Format costs as currency strings for display."""
        return {
            room_id: f"£{cost:.0f}" 
            for room_id, cost in self.allocation.items()
        }
    
    def get_room_cost_deviations(self, house) -> Dict[str, float]:
        """Get deviations from average cost for each room."""
        average_cost = house.target_mean
        return {
            room_id: cost - average_cost
            for room_id, cost in self.allocation.items()
        }


class BaseOptimizer(ABC):
    """
    Abstract base class for all rent optimization algorithms.
    Improved for Streamlit compatibility with serialization support.
    """
    
    def __init__(self, name: str, **parameters):
        """
        Initialize the optimizer.
        
        Args:
            name: Name of the optimization method
            **parameters: Method-specific parameters
        """
        self.name = name
        self.parameters = parameters
        # Don't store house reference - pass it to methods instead
    
    @abstractmethod
    def optimize(self, house) -> Dict[str, float]:
        """
        Perform the optimization and return the allocation.
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        pass
    
    def get_optimization_result(self, house) -> OptimizationResult:
        """
        Get complete optimization result with metrics.
        
        Args:
            house: House object to optimize
            
        Returns:
            OptimizationResult object with all data
        """
        allocation = self.optimize(house)
        fairness_metrics = self.calculate_fairness_metrics(house, allocation)
        
        return OptimizationResult(
            method_name=self.name,
            allocation=allocation,
            fairness_metrics=fairness_metrics,
            parameters=self.parameters
        )
    
    def get_result_dataframe(self, house, allocation: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Get optimization results as a DataFrame.
        
        Args:
            house: House object
            allocation: Pre-calculated allocation or None to calculate
            
        Returns:
            DataFrame with detailed allocation results
        """
        if allocation is None:
            allocation = self.optimize(house)
        
        data = []
        for room in house.individual_rooms:
            room_id = room.room_id
            total_cost = allocation.get(room_id, 0)
            individual_cost = total_cost - house.shared_cost_per_person
            deviation_from_mean = total_cost - house.target_mean
            
            # Calculate deviation from proportional
            proportional_allocation = house.get_proportional_allocation()
            proportional_cost = proportional_allocation.get(room_id, 0)
            deviation_from_proportional = total_cost - proportional_cost
            
            data.append({
                'Method': self.name,
                'Room_ID': room_id,
                'Size_sqm': room.size,
                'Desirability_Score': round(room.desirability_score or 0, 2),
                'Individual_Cost': round(individual_cost, 2),
                'Shared_Cost_Portion': round(house.shared_cost_per_person, 2),
                'Total_Monthly_Cost': round(total_cost, 2),
                'Deviation_from_Mean': round(deviation_from_mean, 2),
                'Deviation_from_Proportional': round(deviation_from_proportional, 2),
                'Cost_per_sqm': round(total_cost / room.size, 2) if room.size > 0 else 0
            })
        
        return pd.DataFrame(data)
    
    def get_display_table_data(self, house, allocation: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Get optimization results formatted for UI table display.
        
        Args:
            house: House object
            allocation: Pre-calculated allocation or None to calculate
            
        Returns:
            List of dictionaries with formatted data for UI display
        """
        if allocation is None:
            allocation = self.optimize(house)
        
        data = []
        for room in house.individual_rooms:
            room_id = room.room_id
            total_cost = allocation.get(room_id, 0)
            individual_cost = total_cost - house.shared_cost_per_person
            deviation_from_mean = total_cost - house.target_mean
            
            data.append({
                'Room': room_id,
                'Size (sqm)': f"{room.size:.1f}",
                'Desirability': f"{room.desirability_score:.1f}/5",
                'Monthly Rent': f"£{total_cost:.0f}",
                'vs. Average': f"{deviation_from_mean:+.0f}",
                'Per sqm': f"£{total_cost/room.size:.1f}"
            })
        
        return data
    
    def get_room_cards_data(self, house, allocation: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Get data formatted for room cards in the UI.
        
        Args:
            house: House object
            allocation: Pre-calculated allocation or None to calculate
            
        Returns:
            List of dictionaries with room card data
        """
        if allocation is None:
            allocation = self.optimize(house)
        
        cards_data = []
        for room in house.individual_rooms:
            room_id = room.room_id
            total_cost = allocation.get(room_id, 0)
            deviation_from_mean = total_cost - house.target_mean
            
            # Determine color based on cost relative to average
            if deviation_from_mean > 50:
                color = "#e74c3c"  # Red for expensive
                status = "expensive"
            elif deviation_from_mean < -50:
                color = "#27ae60"  # Green for cheap
                status = "cheap"
            else:
                color = "#3498db"  # Blue for average
                status = "average"
            
            cards_data.append({
                'room_id': room_id,
                'total_cost': total_cost,
                'cost_formatted': f"£{total_cost:.0f}",
                'size': room.size,
                'size_formatted': f"{room.size:.1f}m²",
                'deviation_from_mean': deviation_from_mean,
                'deviation_formatted': f"{deviation_from_mean:+.0f}",
                'cost_per_sqm': total_cost / room.size,
                'color': color,
                'status': status
            })
        
        return cards_data
    
    def calculate_fairness_metrics(self, house, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate various fairness metrics for the allocation.
        
        Args:
            house: House object
            allocation: Cost allocation
            
        Returns:
            Dictionary of fairness metrics
        """
        cost_values = list(allocation.values())
        proportional_costs = house.get_proportional_allocation()
        
        # Validation check
        total_allocation = sum(cost_values)
        rent_error = abs(total_allocation - house.total_rent)
        
        # Deviation from target mean
        mean_deviations = [abs(cost - house.target_mean) for cost in cost_values]
        avg_deviation_from_mean = np.mean(mean_deviations)
        max_deviation_from_mean = max(mean_deviations)
        
        # Deviation from proportional
        prop_deviations = [abs(allocation[room_id] - proportional_costs[room_id]) 
                          for room_id in allocation.keys() if room_id in proportional_costs]
        avg_deviation_from_prop = np.mean(prop_deviations) if prop_deviations else 0
        max_deviation_from_prop = max(prop_deviations) if prop_deviations else 0
        
        # Statistical measures
        if len(cost_values) > 1:
            variance = np.var(cost_values, ddof=1)  # Sample variance
            std_dev = np.std(cost_values, ddof=1)   # Sample std dev
        else:
            variance = 0
            std_dev = 0
            
        cost_range = max(cost_values) - min(cost_values)
        mean_cost = np.mean(cost_values)
        
        # Coefficient of variation (normalized measure of dispersion)
        cv = std_dev / mean_cost if mean_cost > 0 else 0
        
        # Gini coefficient (measure of inequality)
        gini = self._calculate_gini_coefficient(cost_values)
        
        # Fairness score (0-100, higher is better)
        # Composite score considering multiple factors
        fairness_score = self._calculate_fairness_score(
            cost_values, house.target_mean, std_dev, gini
        )
        
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
            'Fairness_Score': round(fairness_score, 1),
            'Total_Rent_Check': round(total_allocation, 2),
            'Rent_Error': round(rent_error, 2)
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate the Gini coefficient for measuring inequality."""
        if len(values) <= 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumulative_values = np.cumsum(sorted_values)
        
        # Handle edge case where all values are the same
        if cumulative_values[-1] == 0:
            return 0.0
            
        return (n + 1 - 2 * sum(cumulative_values) / cumulative_values[-1]) / n
    
    def _calculate_fairness_score(self, 
                                 cost_values: List[float], 
                                 target_mean: float, 
                                 std_dev: float, 
                                 gini: float) -> float:
        """
        Calculate a composite fairness score (0-100, higher = more fair).
        
        Considers:
        - Low standard deviation (consistency)
        - Low Gini coefficient (equality)
        - Reasonable spread around mean
        """
        if len(cost_values) <= 1:
            return 100.0
        
        # Normalize standard deviation (as % of mean)
        mean_cost = np.mean(cost_values)
        if mean_cost > 0:
            normalized_std = std_dev / mean_cost
        else:
            normalized_std = 0
        
        # Score components (0-1 scale, higher is better)
        # Standard deviation score: penalize high variation
        std_score = max(0, 1 - normalized_std * 2)  # Good if std < 50% of mean
        
        # Gini score: penalize inequality
        gini_score = max(0, 1 - gini * 4)  # Good if Gini < 0.25
        
        # Range score: penalize extreme spreads
        cost_range = max(cost_values) - min(cost_values)
        normalized_range = cost_range / target_mean if target_mean > 0 else 0
        range_score = max(0, 1 - normalized_range * 0.5)  # Good if range < 200% of mean
        
        # Weighted composite score
        fairness_score = (
            std_score * 0.4 +      # 40% weight on consistency
            gini_score * 0.4 +     # 40% weight on equality
            range_score * 0.2      # 20% weight on reasonable range
        ) * 100
        
        return max(0, min(100, fairness_score))
    
    def validate_allocation(self, house, allocation: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        Validate that the allocation is correct.
        
        Args:
            house: House object
            allocation: Cost allocation to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if allocation exists
        if not allocation:
            errors.append("Allocation is empty")
            return False, errors
        
        # Check that all rooms are included
        expected_rooms = {room.room_id for room in house.individual_rooms}
        allocation_rooms = set(allocation.keys())
        
        missing_rooms = expected_rooms - allocation_rooms
        extra_rooms = allocation_rooms - expected_rooms
        
        if missing_rooms:
            errors.append(f"Missing rooms in allocation: {missing_rooms}")
        if extra_rooms:
            errors.append(f"Extra rooms in allocation: {extra_rooms}")
        
        # Check that costs are positive
        negative_costs = [room_id for room_id, cost in allocation.items() if cost < 0]
        if negative_costs:
            errors.append(f"Negative costs for rooms: {negative_costs}")
        
        # Check total rent
        total = sum(allocation.values())
        if abs(total - house.total_rent) > 0.01:
            errors.append(f"Total allocation (£{total:.2f}) doesn't match total rent (£{house.total_rent:.2f})")
        
        # Check individual costs aren't negative
        for room in house.individual_rooms:
            if room.room_id in allocation:
                individual_cost = allocation[room.room_id] - house.shared_cost_per_person
                if individual_cost < -0.01:  # Small tolerance for rounding
                    errors.append(f"Individual cost for {room.room_id} is negative: £{individual_cost:.2f}")
        
        return len(errors) == 0, errors
    
    def print_summary(self, house):
        """Print a summary of the optimization results."""
        allocation = self.optimize(house)
        
        print(f"\n💰 {self.name.upper()} OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Print parameters if any
        if self.parameters:
            print("Parameters:")
            for key, value in self.parameters.items():
                print(f"  {key}: {value}")
            print()
        
        for room in house.individual_rooms:
            room_id = room.room_id
            total_cost = allocation[room_id]
            individual_cost = total_cost - house.shared_cost_per_person
            deviation = total_cost - house.target_mean
            
            print(f"{room_id} ({room.size} sqm):")
            print(f"  Individual room: £{individual_cost:.2f}")
            print(f"  Shared portion:  £{house.shared_cost_per_person:.2f}")
            print(f"  TOTAL MONTHLY:   £{total_cost:.2f} ({deviation:+.0f} from £{house.target_mean:.0f} mean)")
            if room.desirability_score:
                print(f"  Desirability:    {room.desirability_score:.1f}/5")
            print()
        
        # Validation and metrics
        is_valid, errors = self.validate_allocation(house, allocation)
        if is_valid:
            print("✅ Allocation is valid")
        else:
            print("❌ Allocation has errors:")
            for error in errors:
                print(f"  - {error}")
        
        metrics = self.calculate_fairness_metrics(house, allocation)
        print(f"📊 Fairness Score: {metrics['Fairness_Score']}/100")
        print(f"📊 Range: £{metrics['Range_Max_Min']} | Std Dev: £{metrics['Standard_Deviation']:.2f} | Gini: {metrics['Gini_Coefficient']:.3f}")
    
    def get_costs_summary(self, house, allocation: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of costs broken down by room.
        
        Args:
            house: House object
            allocation: Pre-calculated allocation or None to calculate
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        if allocation is None:
            allocation = self.optimize(house)
        
        summary = {}
        for room in house.individual_rooms:
            room_id = room.room_id
            total_cost = allocation[room_id]
            individual_cost = total_cost - house.shared_cost_per_person
            
            summary[room_id] = {
                'size': room.size,
                'individual_cost': round(individual_cost, 2),
                'shared_cost': round(house.shared_cost_per_person, 2),
                'total_cost': round(total_cost, 2),
                'cost_per_sqm': round(total_cost / room.size, 2) if room.size > 0 else 0,
                'desirability': round(room.desirability_score or 0, 2)
            }
        
        return summary
    
    def get_export_data(self, house, allocation: Optional[Dict[str, float]] = None) -> Dict:
        """
        Get comprehensive data formatted for export.
        
        Args:
            house: House object  
            allocation: Pre-calculated allocation or None to calculate
            
        Returns:
            Dictionary with all export data
        """
        if allocation is None:
            allocation = self.optimize(house)
        
        result = self.get_optimization_result(house)
        
        return {
            'method_info': {
                'name': self.name,
                'parameters': self.parameters
            },
            'house_summary': house.get_house_summary(),
            'allocation': allocation,
            'room_details': self.get_display_table_data(house, allocation),
            'fairness_metrics': result.fairness_metrics,
            'cost_summary': self.get_costs_summary(house, allocation)
        }
    
    def to_dict(self) -> Dict:
        """Convert optimizer to dictionary for JSON serialization."""
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BaseOptimizer':
        """
        Create optimizer from dictionary.
        Note: This is a factory method that should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement from_dict method")
    
    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}(name='{self.name}', {params_str})"


# Streamlit Integration Helper
class OptimizerManager:
    """Helper class for managing optimizers in Streamlit."""
    
    @staticmethod
    def save_results_to_session_state(results: List[OptimizationResult], key: str = "optimization_results"):
        """Save optimization results to Streamlit session state."""
        import streamlit as st
        st.session_state[key] = [result.to_dict() for result in results]
    
    @staticmethod
    def load_results_from_session_state(key: str = "optimization_results") -> List[OptimizationResult]:
        """Load optimization results from Streamlit session state."""
        import streamlit as st
        if key in st.session_state:
            try:
                return [OptimizationResult.from_dict(data) for data in st.session_state[key]]
            except Exception as e:
                st.error(f"Error loading optimization results: {e}")
                return []
        return []
    
    @staticmethod
    def create_comparison_dataframe(results: List[OptimizationResult], house) -> pd.DataFrame:
        """Create a comparison DataFrame from multiple optimization results."""
        all_data = []
        
        for result in results:
            # Get the detailed dataframe for this method
            optimizer = BaseOptimizer(result.method_name, **result.parameters)
            df = optimizer.get_result_dataframe(house, result.allocation)
            all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def get_fairness_comparison(results: List[OptimizationResult]) -> pd.DataFrame:
        """Create a DataFrame comparing fairness metrics across methods."""
        fairness_data = []
        
        for result in results:
            fairness_data.append(result.fairness_metrics)
        
        if fairness_data:
            return pd.DataFrame(fairness_data)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def create_range_comparison_data(house, spread_percentages: List[float]) -> pd.DataFrame:
        """
        Create comparison data for multiple spread percentages.
        
        Args:
            house: House object
            spread_percentages: List of spread percentages to compare
            
        Returns:
            DataFrame with comparison data
        """
        from ..optimizers.quadratic_optimizer import QuadraticOptimizer
        
        comparison_data = []
        
        for spread_percent in spread_percentages:
            optimizer = QuadraticOptimizer.from_spread_percentage(spread_percent)
            result = optimizer.get_optimization_result(house)
            
            for room in house.individual_rooms:
                cost = result.allocation[room.room_id]
                comparison_data.append({
                    'Spread': f"{spread_percent}%",
                    'Spread_Value': spread_percent,
                    'Room': room.room_id,
                    'Cost': cost,
                    'Size': room.size,
                    'Desirability': room.desirability_score
                })
        
        return pd.DataFrame(comparison_data)