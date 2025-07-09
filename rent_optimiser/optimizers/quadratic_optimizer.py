from typing import Dict
import numpy as np
from .base_optimizer import BaseOptimizer


class QuadraticOptimizer(BaseOptimizer):
    """
    Quadratic programming optimizer that balances equality with proportionality.
    
    Minimizes: Σ(cost_i - target_mean)² + λ * Σ(cost_i - proportional_i)²
    
    λ = 0: Pure equality (everyone pays target_mean)
    λ = ∞: Pure proportional allocation  
    λ = 5: Balanced approach (default)
    """
    
    def __init__(self, lambda_param: float = 5.0):
        """
        Initialize the quadratic optimizer.
        
        Args:
            lambda_param: Weight for proportional deviation penalty (0-10 scale)
        """
        super().__init__(f"Quadratic (λ={lambda_param})", lambda_param=lambda_param)
        self.lambda_param = lambda_param
    
    def optimize(self, house) -> Dict[str, float]:
        """
        Perform quadratic optimization using analytical solution.
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        room_ids = [room.room_id for room in house.individual_rooms]
        
        # Get proportional costs based on desirability-weighted size
        proportional_costs = self._get_weighted_proportional_allocation(house)
        proportional_values = [proportional_costs[room_id] for room_id in room_ids]
        
        # Analytical solution for quadratic optimization:
        # cost_i = (target_mean + λ * proportional_i) / (1 + λ)
        preliminary_costs = [
            (house.target_mean + self.lambda_param * prop_cost) / (1 + self.lambda_param)
            for prop_cost in proportional_values
        ]
        
        # Adjust to ensure total equals total_rent (handles rounding errors)
        preliminary_total = sum(preliminary_costs)
        adjustment = (house.total_rent - preliminary_total) / house.num_people
        
        optimized_costs = [cost + adjustment for cost in preliminary_costs]
        
        # Create allocation dictionary
        allocation = dict(zip(room_ids, optimized_costs))
        
        return allocation
    
    def _get_weighted_proportional_allocation(self, house) -> Dict[str, float]:
        """
        Get proportional allocation based on desirability-weighted room sizes.
        
        This replaces the simple size-based proportion with a weighted approach
        that considers the full desirability score (size + noise + accessibility).
        
        Args:
            house: House object
            
        Returns:
            Dictionary of {room_id: proportional_cost}
        """
        # Calculate weighted areas for individual rooms
        weighted_areas = {}
        total_weighted_area = 0
        
        for room in house.individual_rooms:
            # Use desirability score as a multiplier for effective "room value"
            # Higher desirability = higher proportional share
            weighted_area = room.size * room.desirability_score
            weighted_areas[room.room_id] = weighted_area
            total_weighted_area += weighted_area
        
        # Calculate proportional allocation from the individual room pool
        individual_pool = house.total_rent - (house.shared_cost_per_person * house.num_people)
        
        proportional_allocation = {}
        for room_id, weighted_area in weighted_areas.items():
            proportion = weighted_area / total_weighted_area
            individual_cost = proportion * individual_pool
            total_cost = individual_cost + house.shared_cost_per_person
            proportional_allocation[room_id] = total_cost
        
        return proportional_allocation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuadraticOptimizer':
        """Create optimizer from dictionary for JSON deserialization."""
        lambda_param = data.get('parameters', {}).get('lambda_param', 5.0)
        return cls(lambda_param=lambda_param)
    
    def to_dict(self) -> Dict:
        """Convert optimizer to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict['parameters'] = {'lambda_param': self.lambda_param}
        return base_dict
    
    def get_lambda_description(self) -> str:
        """Get human-readable description of lambda value."""
        if self.lambda_param == 0:
            return "Maximum Equality (everyone pays the same)"
        elif self.lambda_param <= 2:
            return "Strong Equality Preference"
        elif self.lambda_param <= 5:
            return "Balanced Approach" 
        elif self.lambda_param <= 8:
            return "Strong Fairness Preference"
        else:
            return "Maximum Fairness (pure proportional)"