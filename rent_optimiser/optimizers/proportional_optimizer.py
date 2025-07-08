from typing import Dict
from .base_optimizer import BaseOptimizer
from house import House


class ProportionalOptimizer(BaseOptimizer):
    """
    Optimizer that allocates rent proportionally based on room size.
    
    This is the baseline method where larger rooms pay proportionally more.
    """
    
    def __init__(self, house: House):
        """Initialize the proportional optimizer."""
        super().__init__(house, "Proportional")
    
    def optimize(self) -> Dict[str, float]:
        """
        Calculate proportional allocation based on room sizes.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        # Use the house's built-in proportional calculation
        self.allocation = self.house.get_proportional_allocation()
        return self.allocation


class DesirabilityWeightedOptimizer(BaseOptimizer):
    """
    Optimizer that combines room size with desirability scores.
    
    Allows weighting between pure size-based allocation and desirability-adjusted allocation.
    """
    
    def __init__(self, house: House, desirability_weight: float = 0.3):
        """
        Initialize the desirability-weighted optimizer.
        
        Args:
            house: House object
            desirability_weight: Weight for desirability (0-1). 
                               0 = pure size, 1 = pure desirability
        """
        super().__init__(house, f"Size+Desirability({desirability_weight*100:.0f}%)")
        self.desirability_weight = desirability_weight
    
    def optimize(self) -> Dict[str, float]:
        """
        Calculate allocation based on both room size and desirability.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        # Get desirability scores
        desirability_scores = {room.room_id: room.desirability_score or 5.0 
                              for room in self.house.individual_rooms}
        
        if not any(score != 5.0 for score in desirability_scores.values()):
            # If no desirability scores available, fall back to proportional
            return self.house.get_proportional_allocation()
        
        # Normalize desirability scores to 0-1 range
        max_desirability = max(desirability_scores.values())
        min_desirability = min(desirability_scores.values())
        desirability_range = max_desirability - min_desirability or 1
        
        total_weighted_value = 0
        weighted_values = {}
        
        # Calculate weighted value for each room
        for room in self.house.individual_rooms:
            desirability = desirability_scores[room.room_id]
            normalized_desirability = (desirability - min_desirability) / desirability_range
            
            # Combine size and desirability
            size_weight = 1 - self.desirability_weight
            weighted_value = (size_weight * room.size + 
                            self.desirability_weight * normalized_desirability * (room.size * 0.5))
            
            weighted_values[room.room_id] = weighted_value
            total_weighted_value += weighted_value
        
        # Allocate costs based on weighted values
        allocation = {}
        for room in self.house.individual_rooms:
            weighted_value = weighted_values[room.room_id]
            room_cost = (weighted_value / total_weighted_value) * self.house.individual_rent_pool
            total_cost = room_cost + self.house.shared_cost_per_person
            allocation[room.room_id] = total_cost
        
        self.allocation = allocation
        return allocation


class EqualPaymentOptimizer(BaseOptimizer):
    """
    Optimizer where everyone pays exactly the same amount.
    
    This represents perfect equality but ignores room size differences.
    """
    
    def __init__(self, house: House):
        """Initialize the equal payment optimizer."""
        super().__init__(house, "Equal Payment")
    
    def optimize(self) -> Dict[str, float]:
        """
        Calculate equal payment for all rooms.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        equal_cost = self.house.target_mean
        
        self.allocation = {room.room_id: equal_cost 
                          for room in self.house.individual_rooms}
        
        return self.allocation