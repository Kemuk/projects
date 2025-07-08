from typing import Dict, Any
from .base_optimizer import BaseOptimizer  # Updated import


class ProportionalOptimizer(BaseOptimizer):
    """
    Optimizer that allocates rent proportionally based on room size.
    
    This is the baseline method where larger rooms pay proportionally more.
    The simplest and most straightforward allocation method.
    """
    
    def __init__(self):
        """Initialize the proportional optimizer."""
        super().__init__(name="Proportional by Size")
    
    def optimize(self, house) -> Dict[str, float]:
        """
        Calculate proportional allocation based on room sizes.
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        # Validate inputs
        if not house.individual_rooms:
            raise ValueError("House must have individual rooms")
        
        total_individual_area = sum(room.size for room in house.individual_rooms)
        if total_individual_area <= 0:
            raise ValueError("Total room area must be positive")
        
        # Calculate proportional allocation
        allocation = {}
        for room in house.individual_rooms:
            # Individual room cost based on size proportion
            room_proportion = room.size / total_individual_area
            individual_cost = room_proportion * house.individual_rent_pool
            
            # Add shared cost portion
            total_cost = individual_cost + house.shared_cost_per_person
            allocation[room.room_id] = total_cost
        
        return allocation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProportionalOptimizer':
        """Create ProportionalOptimizer from dictionary."""
        return cls()
    
    def __repr__(self):
        return "ProportionalOptimizer()"


class DesirabilityWeightedOptimizer(BaseOptimizer):
    """
    Optimizer that combines room size with desirability scores.
    
    Allows weighting between pure size-based allocation and desirability-adjusted allocation.
    Higher desirability rooms pay less, lower desirability rooms pay more.
    """
    
    def __init__(self, desirability_weight: float = 0.3):
        """
        Initialize the desirability-weighted optimizer.
        
        Args:
            desirability_weight: Weight for desirability adjustment (0-1). 
                               0 = pure size-based allocation
                               1 = maximum desirability adjustment
        """
        if not 0 <= desirability_weight <= 1:
            raise ValueError("Desirability weight must be between 0 and 1")
        
        super().__init__(
            name=f"Size + Desirability ({desirability_weight*100:.0f}%)",
            desirability_weight=desirability_weight
        )
    
    def optimize(self, house) -> Dict[str, float]:
        """
        Calculate allocation based on both room size and desirability.
        
        Logic:
        1. Start with proportional allocation by size
        2. Apply desirability adjustments (better rooms pay less)
        3. Ensure total still equals target rent
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        desirability_weight = self.parameters['desirability_weight']
        
        # Validate inputs
        if not house.individual_rooms:
            raise ValueError("House must have individual rooms")
        
        # Get baseline proportional allocation
        proportional_optimizer = ProportionalOptimizer()
        baseline_allocation = proportional_optimizer.optimize(house)
        
        # If no desirability weighting, return proportional
        if desirability_weight == 0:
            return baseline_allocation
        
        # Get desirability scores for all rooms
        desirability_scores = {}
        for room in house.individual_rooms:
            score = room.desirability_score
            if score is None or score <= 0:
                # Default to neutral score if not available
                score = 3.0
            desirability_scores[room.room_id] = score
        
        # Check if all rooms have the same desirability (no need to adjust)
        unique_scores = set(desirability_scores.values())
        if len(unique_scores) <= 1:
            return baseline_allocation
        
        # Calculate desirability adjustments
        avg_desirability = sum(desirability_scores.values()) / len(desirability_scores)
        max_desirability = max(desirability_scores.values())
        min_desirability = min(desirability_scores.values())
        desirability_range = max_desirability - min_desirability
        
        # Apply adjustments to each room
        allocation = {}
        adjustment_total = 0
        
        for room in house.individual_rooms:
            baseline_cost = baseline_allocation[room.room_id]
            desirability = desirability_scores[room.room_id]
            
            # Calculate adjustment factor
            # Positive factor = room pays less (more desirable)
            # Negative factor = room pays more (less desirable)
            if desirability_range > 0:
                desirability_factor = (desirability - avg_desirability) / desirability_range
            else:
                desirability_factor = 0
            
            # Apply adjustment as percentage of baseline cost
            # Max adjustment is ±20% of baseline cost scaled by desirability_weight
            max_adjustment_rate = 0.20 * desirability_weight
            adjustment = -desirability_factor * max_adjustment_rate * baseline_cost
            
            adjusted_cost = baseline_cost + adjustment
            allocation[room.room_id] = adjusted_cost
            adjustment_total += adjustment
        
        # Normalize to ensure total equals target rent
        # Distribute any rounding error proportionally
        actual_total = sum(allocation.values())
        target_total = house.total_rent
        
        if abs(actual_total - target_total) > 0.01:
            correction_factor = target_total / actual_total
            for room_id in allocation:
                allocation[room_id] *= correction_factor
        
        return allocation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DesirabilityWeightedOptimizer':
        """Create DesirabilityWeightedOptimizer from dictionary."""
        desirability_weight = data.get('parameters', {}).get('desirability_weight', 0.3)
        return cls(desirability_weight=desirability_weight)
    
    def __repr__(self):
        weight = self.parameters['desirability_weight']
        return f"DesirabilityWeightedOptimizer(desirability_weight={weight})"


class EqualPaymentOptimizer(BaseOptimizer):
    """
    Optimizer where everyone pays exactly the same amount.
    
    This represents perfect equality but completely ignores room size and desirability differences.
    Useful as a baseline for measuring inequality in other methods.
    """
    
    def __init__(self):
        """Initialize the equal payment optimizer."""
        super().__init__(name="Equal Payment")
    
    def optimize(self, house) -> Dict[str, float]:
        """
        Calculate equal payment for all rooms.
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        # Validate inputs
        if not house.individual_rooms:
            raise ValueError("House must have individual rooms")
        
        equal_cost = house.target_mean
        
        allocation = {room.room_id: equal_cost 
                     for room in house.individual_rooms}
        
        return allocation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EqualPaymentOptimizer':
        """Create EqualPaymentOptimizer from dictionary."""
        return cls()
    
    def __repr__(self):
        return "EqualPaymentOptimizer()"


class FloorAdjustedOptimizer(BaseOptimizer):
    """
    NEW: Optimizer that adjusts for floor level preferences.
    
    Generally, higher floors are more desirable (quieter, better views),
    but ground floor might be preferred for convenience.
    """
    
    def __init__(self, floor_adjustment_rate: float = 0.15):
        """
        Initialize the floor-adjusted optimizer.
        
        Args:
            floor_adjustment_rate: Maximum adjustment rate based on floor (0-1)
        """
        if not 0 <= floor_adjustment_rate <= 1:
            raise ValueError("Floor adjustment rate must be between 0 and 1")
        
        super().__init__(
            name=f"Floor Adjusted ({floor_adjustment_rate*100:.0f}%)",
            floor_adjustment_rate=floor_adjustment_rate
        )
    
    def optimize(self, house) -> Dict[str, float]:
        """
        Calculate allocation with floor-level adjustments.
        
        Args:
            house: House object to optimize
            
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        floor_adjustment_rate = self.parameters['floor_adjustment_rate']
        
        # Get baseline proportional allocation
        proportional_optimizer = ProportionalOptimizer()
        baseline_allocation = proportional_optimizer.optimize(house)
        
        if floor_adjustment_rate == 0:
            return baseline_allocation
        
        # Get floor levels from desirability factors
        floor_levels = {}
        for room in house.individual_rooms:
            if hasattr(room.desirability_factors, 'floor_level'):
                floor_levels[room.room_id] = room.desirability_factors.floor_level
            else:
                floor_levels[room.room_id] = 1  # Default to ground floor
        
        # Calculate floor adjustments
        avg_floor = sum(floor_levels.values()) / len(floor_levels)
        max_floor = max(floor_levels.values())
        min_floor = min(floor_levels.values())
        floor_range = max_floor - min_floor
        
        if floor_range == 0:
            return baseline_allocation
        
        allocation = {}
        for room in house.individual_rooms:
            baseline_cost = baseline_allocation[room.room_id]
            floor_level = floor_levels[room.room_id]
            
            # Higher floors generally pay less (more desirable)
            # But this can be adjusted based on house-specific preferences
            floor_factor = (floor_level - avg_floor) / floor_range
            adjustment = -floor_factor * floor_adjustment_rate * baseline_cost
            
            allocation[room.room_id] = baseline_cost + adjustment
        
        # Normalize to target total
        actual_total = sum(allocation.values())
        target_total = house.total_rent
        
        if abs(actual_total - target_total) > 0.01:
            correction_factor = target_total / actual_total
            for room_id in allocation:
                allocation[room_id] *= correction_factor
        
        return allocation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FloorAdjustedOptimizer':
        """Create FloorAdjustedOptimizer from dictionary."""
        floor_adjustment_rate = data.get('parameters', {}).get('floor_adjustment_rate', 0.15)
        return cls(floor_adjustment_rate=floor_adjustment_rate)
    
    def __repr__(self):
        rate = self.parameters['floor_adjustment_rate']
        return f"FloorAdjustedOptimizer(floor_adjustment_rate={rate})"


# Factory function for creating optimizers from saved data
def create_optimizer_from_dict(data: Dict) -> BaseOptimizer:
    """
    Factory function to create optimizers from saved dictionary data.
    
    Args:
        data: Dictionary containing optimizer class and parameters
        
    Returns:
        Instantiated optimizer object
    """
    class_name = data.get('class_name')
    
    optimizer_classes = {
        'ProportionalOptimizer': ProportionalOptimizer,
        'DesirabilityWeightedOptimizer': DesirabilityWeightedOptimizer,
        'EqualPaymentOptimizer': EqualPaymentOptimizer,
        'FloorAdjustedOptimizer': FloorAdjustedOptimizer,
    }
    
    if class_name not in optimizer_classes:
        raise ValueError(f"Unknown optimizer class: {class_name}")
    
    return optimizer_classes[class_name].from_dict(data)


# Streamlit helper functions
def get_available_optimizers() -> Dict[str, type]:
    """Get dictionary of available optimizer classes for Streamlit UI."""
    return {
        "Proportional by Size": ProportionalOptimizer,
        "Size + Desirability": DesirabilityWeightedOptimizer,
        "Equal Payment": EqualPaymentOptimizer,
        "Floor Level Adjusted": FloorAdjustedOptimizer,
    }


def create_optimizer_from_streamlit_form(method_name: str, parameters: Dict[str, Any]) -> BaseOptimizer:
    """
    Create optimizer from Streamlit form inputs.
    
    Args:
        method_name: Name of the optimization method
        parameters: Dictionary of method-specific parameters
        
    Returns:
        Configured optimizer instance
    """
    if method_name == "Proportional by Size":
        return ProportionalOptimizer()
    
    elif method_name == "Size + Desirability":
        weight = parameters.get('desirability_weight', 0.3)
        return DesirabilityWeightedOptimizer(desirability_weight=weight)
    
    elif method_name == "Equal Payment":
        return EqualPaymentOptimizer()
    
    elif method_name == "Floor Level Adjusted":
        rate = parameters.get('floor_adjustment_rate', 0.15)
        return FloorAdjustedOptimizer(floor_adjustment_rate=rate)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")