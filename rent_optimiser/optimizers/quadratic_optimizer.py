from typing import Dict
import numpy as np
from scipy.optimize import minimize
from .base_optimizer import BaseOptimizer
from house import House


class QuadraticOptimizer(BaseOptimizer):
    """
    Quadratic programming optimizer that balances equality with proportionality.
    
    Minimizes: Σ(cost_i - target_mean)² + λ * Σ(cost_i - proportional_i)²
    
    λ = 0: Pure equality (everyone pays target_mean)
    λ = ∞: Pure proportional allocation  
    λ = 1: Balanced approach
    """
    
    def __init__(self, house: House, lambda_param: float = 1.0):
        """
        Initialize the quadratic optimizer.
        
        Args:
            house: House object
            lambda_param: Weight for proportional deviation penalty
        """
        super().__init__(house, f"Quadratic(λ={lambda_param})")
        self.lambda_param = lambda_param
    
    def optimize(self) -> Dict[str, float]:
        """
        Perform quadratic optimization.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        room_ids = [room.room_id for room in self.house.individual_rooms]
        proportional_costs = self.house.get_proportional_allocation()
        proportional_values = [proportional_costs[room_id] for room_id in room_ids]
        
        # For quadratic objective with linear constraint, analytical solution exists:
        # cost_i = (target_mean + λ * proportional_i) / (1 + λ)
        preliminary_costs = [
            (self.house.target_mean + self.lambda_param * prop_cost) / (1 + self.lambda_param)
            for prop_cost in proportional_values
        ]
        
        # Adjust to ensure total equals total_rent
        preliminary_total = sum(preliminary_costs)
        adjustment = (self.house.total_rent - preliminary_total) / self.house.num_people
        
        optimized_costs = [cost + adjustment for cost in preliminary_costs]
        
        self.allocation = dict(zip(room_ids, optimized_costs))
        return self.allocation


class ProgressiveCapOptimizer(BaseOptimizer):
    """
    Progressive cap optimizer that limits deviations from the mean.
    
    Caps how much anyone can pay above or below the target mean,
    then redistributes to maintain total rent.
    """
    
    def __init__(self, house: House, max_above_mean: float = 100, max_below_mean: float = 150):
        """
        Initialize the progressive cap optimizer.
        
        Args:
            house: House object
            max_above_mean: Maximum amount above target mean (£)
            max_below_mean: Maximum amount below target mean (£)
        """
        super().__init__(house, f"Progressive Cap(+{max_above_mean}/-{max_below_mean})")
        self.max_above_mean = max_above_mean
        self.max_below_mean = max_below_mean
    
    def optimize(self) -> Dict[str, float]:
        """
        Apply progressive caps and redistribute.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        proportional_costs = self.house.get_proportional_allocation()
        
        # Apply caps to proportional costs
        capped_costs = {}
        for room_id, prop_cost in proportional_costs.items():
            max_cost = self.house.target_mean + self.max_above_mean
            min_cost = self.house.target_mean - self.max_below_mean
            capped_costs[room_id] = max(min_cost, min(max_cost, prop_cost))
        
        # Redistribute to maintain total
        capped_total = sum(capped_costs.values())
        redistribution = (self.house.total_rent - capped_total) / self.house.num_people
        
        self.allocation = {room_id: cost + redistribution 
                          for room_id, cost in capped_costs.items()}
        
        return self.allocation


class MinMaxDeviationOptimizer(BaseOptimizer):
    """
    Minimizes the maximum deviation from target mean (Rawlsian approach).
    
    Uses scipy.optimize to find the solution that minimizes the worst-off person's deviation.
    """
    
    def __init__(self, house: House):
        """Initialize the min-max deviation optimizer."""
        super().__init__(house, "Min Max Deviation")
    
    def optimize(self) -> Dict[str, float]:
        """
        Minimize the maximum deviation from target mean.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        room_ids = [room.room_id for room in self.house.individual_rooms]
        n = len(room_ids)
        
        def objective(costs):
            # Minimize maximum absolute deviation from target mean
            deviations = [abs(cost - self.house.target_mean) for cost in costs]
            return max(deviations)
        
        def constraint(costs):
            # Total must equal total_rent
            return sum(costs) - self.house.total_rent
        
        # Initial guess: proportional allocation
        proportional_costs = self.house.get_proportional_allocation()
        initial_guess = [proportional_costs[room_id] for room_id in room_ids]
        
        # Constraints
        constraints = {'type': 'eq', 'fun': constraint}
        
        # Bounds: costs must be positive and reasonable
        bounds = [(100, 2000) for _ in range(n)]
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                         constraints=constraints, bounds=bounds)
        
        if result.success:
            self.allocation = dict(zip(room_ids, result.x))
        else:
            print(f"Optimization failed for {self.name}, falling back to proportional")
            self.allocation = proportional_costs
        
        return self.allocation


class CustomWeightedOptimizer(BaseOptimizer):
    """
    Custom optimizer that allows manual weighting of different factors.
    
    Combines size, desirability, and custom adjustments with user-defined weights.
    """
    
    def __init__(self, house: House, 
                 size_weight: float = 0.7,
                 desirability_weight: float = 0.2,
                 custom_adjustments: Dict[str, float] = None):
        """
        Initialize the custom weighted optimizer.
        
        Args:
            house: House object
            size_weight: Weight for room size factor (0-1)
            desirability_weight: Weight for desirability factor (0-1)
            custom_adjustments: Manual adjustments per room {room_id: adjustment_factor}
        """
        super().__init__(house, f"Custom Weighted(S:{size_weight:.1f},D:{desirability_weight:.1f})")
        self.size_weight = size_weight
        self.desirability_weight = desirability_weight
        self.custom_weight = 1.0 - size_weight - desirability_weight
        self.custom_adjustments = custom_adjustments or {}
        
        # Validate weights
        if self.size_weight + self.desirability_weight > 1.0:
            raise ValueError("Size weight + desirability weight cannot exceed 1.0")
    
    def optimize(self) -> Dict[str, float]:
        """
        Calculate allocation using custom weights.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost}
        """
        total_weighted_value = 0
        weighted_values = {}
        
        # Get normalized factors
        sizes = {room.room_id: room.size for room in self.house.individual_rooms}
        max_size = max(sizes.values())
        normalized_sizes = {room_id: size / max_size for room_id, size in sizes.items()}
        
        desirability_scores = {room.room_id: room.desirability_score or 5.0 
                              for room in self.house.individual_rooms}
        max_desirability = max(desirability_scores.values())
        min_desirability = min(desirability_scores.values())
        desirability_range = max_desirability - min_desirability or 1
        normalized_desirability = {
            room_id: (score - min_desirability) / desirability_range
            for room_id, score in desirability_scores.items()
        }
        
        # Calculate weighted values
        for room in self.house.individual_rooms:
            room_id = room.room_id
            
            size_component = self.size_weight * normalized_sizes[room_id]
            desirability_component = self.desirability_weight * normalized_desirability[room_id]
            custom_component = self.custom_weight * self.custom_adjustments.get(room_id, 0.5)
            
            weighted_value = size_component + desirability_component + custom_component
            weighted_values[room_id] = weighted_value
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
