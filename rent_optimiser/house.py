from typing import List, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import json


@dataclass
class RoomConstraint:
    """Constraint on acceptable rent range for a specific room."""
    room_id: str
    min_acceptable: Optional[float] = None  # Minimum acceptable monthly rent
    max_acceptable: Optional[float] = None  # Maximum acceptable monthly rent
    is_active: bool = False  # Whether this constraint is currently enforced
    
    def __post_init__(self):
        """Validate constraint after initialization."""
        if self.min_acceptable is not None and self.max_acceptable is not None:
            if self.min_acceptable > self.max_acceptable:
                raise ValueError(f"Min acceptable ({self.min_acceptable}) cannot be greater than max acceptable ({self.max_acceptable})")
    
    def is_satisfied_by(self, rent_amount: float) -> bool:
        """Check if a rent amount satisfies this constraint."""
        if not self.is_active:
            return True
        
        if self.min_acceptable is not None and rent_amount < self.min_acceptable:
            return False
        
        if self.max_acceptable is not None and rent_amount > self.max_acceptable:
            return False
        
        return True
    
    def get_violation_description(self, rent_amount: float) -> Optional[str]:
        """Get description of constraint violation, if any."""
        if not self.is_active or self.is_satisfied_by(rent_amount):
            return None
        
        if self.min_acceptable is not None and rent_amount < self.min_acceptable:
            return f"£{rent_amount:.0f} is below minimum acceptable (£{self.min_acceptable:.0f})"
        
        if self.max_acceptable is not None and rent_amount > self.max_acceptable:
            return f"£{rent_amount:.0f} is above maximum acceptable (£{self.max_acceptable:.0f})"
        
        return "Unknown constraint violation"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'room_id': self.room_id,
            'min_acceptable': self.min_acceptable,
            'max_acceptable': self.max_acceptable,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RoomConstraint':
        """Create from dictionary."""
        return cls(
            room_id=data['room_id'],
            min_acceptable=data.get('min_acceptable'),
            max_acceptable=data.get('max_acceptable'),
            is_active=data.get('is_active', False)
        )


@dataclass
class DesirabilityFactors:
    """Simplified desirability factors for quadratic optimization."""
    size_score: float = 3.0      # 1-5 scale (manually set)
    noise_level: float = 3.0     # 1-5 scale (1=very noisy, 5=very quiet)
    accessibility: float = 3.0   # 1-5 scale (1=hard to reach kitchen/bathroom, 5=very easy)
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted average of the three factors."""
        # Weights: Size 50%, Noise 40%, Accessibility 10%
        weights = {
            'size_score': 0.5,
            'noise_level': 0.4,
            'accessibility': 0.1
        }
        
        score = (
            self.size_score * weights['size_score'] +
            self.noise_level * weights['noise_level'] +
            self.accessibility * weights['accessibility']
        )
        
        return round(score, 2)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'size_score': self.size_score,
            'noise_level': self.noise_level,
            'accessibility': self.accessibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DesirabilityFactors':
        """Create from dictionary for JSON deserialization."""
        return cls(
            size_score=data.get('size_score', 3.0),
            noise_level=data.get('noise_level', 3.0),
            accessibility=data.get('accessibility', 3.0)
        )


class Room:
    """Simplified room class for quadratic optimization."""
    
    def __init__(self, 
                 room_id: str, 
                 size: float, 
                 desirability_factors: Optional[DesirabilityFactors] = None,
                 is_shared: bool = False):
        """
        Initialize a Room object.
        
        Args:
            room_id: Unique identifier for the room
            size: Size of the room in square meters
            desirability_factors: DesirabilityFactors object or None for default
            is_shared: Whether this room is shared among all housemates
        """
        # Validation
        if not room_id or not isinstance(room_id, str):
            raise ValueError("Room ID must be a non-empty string")
        if size <= 0:
            raise ValueError("Room size must be positive")
        
        self.room_id = room_id
        self.size = size
        self.is_shared = is_shared
        self.desirability_factors = desirability_factors or DesirabilityFactors()
        self.desirability_score = self.desirability_factors.calculate_overall_score()
        
        # Cost-related attributes (set by House object)
        self.individual_cost = 0.0
        self.shared_cost_portion = 0.0
        self.total_cost = 0.0
    
    def update_desirability_factors(self, **kwargs):
        """Update specific desirability factors."""
        for key, value in kwargs.items():
            if hasattr(self.desirability_factors, key):
                setattr(self.desirability_factors, key, value)
        self.desirability_score = self.desirability_factors.calculate_overall_score()
    
    def set_costs(self, individual_cost: float, shared_cost_portion: float):
        """Set the costs for this room."""
        if individual_cost < 0 or shared_cost_portion < 0:
            raise ValueError("Costs cannot be negative")
        
        self.individual_cost = individual_cost
        self.shared_cost_portion = shared_cost_portion
        self.total_cost = individual_cost + shared_cost_portion
    
    def get_display_info(self) -> Dict:
        """Get room information formatted for UI display."""
        return {
            'room_id': self.room_id,
            'size': self.size,
            'size_formatted': f"{self.size:.1f}m²",
            'desirability_score': self.desirability_score,
            'desirability_formatted': f"{self.desirability_score:.1f}/5",
            'size_score': self.desirability_factors.size_score,
            'noise_level': self.desirability_factors.noise_level,
            'accessibility': self.desirability_factors.accessibility,
            'total_cost': self.total_cost,
            'individual_cost': self.individual_cost,
            'shared_cost_portion': self.shared_cost_portion,
            'is_shared': self.is_shared
        }
    
    def to_dict(self) -> Dict:
        """Convert room to dictionary representation for JSON serialization."""
        return {
            'room_id': self.room_id,
            'size': self.size,
            'is_shared': self.is_shared,
            'desirability_factors': self.desirability_factors.to_dict(),
            'individual_cost': self.individual_cost,
            'shared_cost_portion': self.shared_cost_portion,
            'total_cost': self.total_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Room':
        """Create Room from dictionary for JSON deserialization."""
        desirability_data = data.get('desirability_factors', {})
        desirability_factors = DesirabilityFactors.from_dict(desirability_data)
        
        room = cls(
            room_id=data['room_id'],
            size=data['size'],
            desirability_factors=desirability_factors,
            is_shared=data.get('is_shared', False)
        )
        
        # Restore cost information if available
        if 'individual_cost' in data:
            room.set_costs(
                data['individual_cost'],
                data.get('shared_cost_portion', 0)
            )
        
        return room


class House:
    """Simplified house class for quadratic optimization."""
    
    def __init__(self, 
                 individual_rooms: List[Room], 
                 shared_room: Optional[Room] = None, 
                 total_rent: float = 0.0):
        """Initialize a House object."""
        # Validation
        if not individual_rooms:
            raise ValueError("House must have at least one individual room")
        if total_rent <= 0:
            raise ValueError("Total rent must be positive")
        
        self.individual_rooms = individual_rooms
        self.shared_room = shared_room
        self.total_rent = total_rent
        
        # Calculated properties
        self.num_people = len(individual_rooms)
        self.total_area = self._calculate_total_area()
        self.target_mean = total_rent / self.num_people
        
        # Initialize room constraints
        self.room_constraints: Dict[str, RoomConstraint] = {}
        for room in individual_rooms:
            self.room_constraints[room.room_id] = RoomConstraint(room_id=room.room_id)
        
        # Calculate costs
        if shared_room:
            self._calculate_shared_costs()
        else:
            self.shared_room_total_cost = 0
            self.shared_cost_per_person = 0
            self.individual_rent_pool = total_rent
    
    def _calculate_total_area(self) -> float:
        """Calculate total area of the house."""
        total = sum(room.size for room in self.individual_rooms)
        if self.shared_room:
            total += self.shared_room.size
        return total
    
    def _calculate_shared_costs(self):
        """Calculate costs related to the shared room."""
        if not self.shared_room:
            return
            
        self.shared_room_total_cost = (self.shared_room.size / self.total_area) * self.total_rent
        self.shared_cost_per_person = self.shared_room_total_cost / self.num_people
        self.individual_rent_pool = self.total_rent - self.shared_room_total_cost
        
        # Set shared room costs
        self.shared_room.set_costs(
            individual_cost=0,
            shared_cost_portion=self.shared_room_total_cost
        )
    
    def get_size_proportions(self) -> Dict[str, float]:
        """Get size as proportion of total individual room area."""
        total_individual_area = sum(room.size for room in self.individual_rooms)
        return {room.room_id: room.size / total_individual_area 
                for room in self.individual_rooms}
    
    def apply_allocation(self, allocation: Dict[str, float]):
        """Apply a new cost allocation to the rooms."""
        if not self.validate_allocation(allocation):
            raise ValueError("Allocation does not sum to total rent")
        
        for room in self.individual_rooms:
            if room.room_id in allocation:
                new_total_cost = allocation[room.room_id]
                new_individual_cost = new_total_cost - self.shared_cost_per_person
                room.set_costs(new_individual_cost, self.shared_cost_per_person)
    
    def validate_allocation(self, allocation: Dict[str, float]) -> bool:
        """Validate that an allocation sums to the total rent."""
        if not allocation:
            return False
        
        # Check that all individual rooms are included
        expected_rooms = {room.room_id for room in self.individual_rooms}
        allocation_rooms = set(allocation.keys())
        if expected_rooms != allocation_rooms:
            return False
        
        # Check that total sums correctly
        total = sum(allocation.values())
        return abs(total - self.total_rent) < 0.01
    
    def get_room_summary_data(self) -> List[Dict]:
        """Get clean room data for UI display."""
        summary = []
        for room in self.individual_rooms:
            summary.append(room.get_display_info())
        return summary
    
    def get_cost_edge_cases(self) -> Dict:
        """Pre-calculate equal split and pure proportional results."""
        # Equal split (everyone pays the same)
        equal_split_cost = self.target_mean
        equal_allocation = {room.room_id: equal_split_cost for room in self.individual_rooms}
        
        # Pure proportional (based on room value: size × desirability)
        room_values = {}
        total_value = 0
        
        for room in self.individual_rooms:
            room_value = room.size * room.desirability_score
            room_values[room.room_id] = room_value
            total_value += room_value
        
        proportional_allocation = {}
        for room_id, room_value in room_values.items():
            proportion = room_value / total_value
            individual_cost = proportion * self.individual_rent_pool
            total_cost = individual_cost + self.shared_cost_per_person
            proportional_allocation[room_id] = total_cost
        
        return {
            'equal_split': equal_allocation,
            'proportional': proportional_allocation,
            'room_values': room_values,
            'total_value': total_value
        }
    
    @property
    def room_rent_range(self) -> Tuple[float, float]:
        """Get min/max possible rent across all methods."""
        edge_cases = self.get_cost_edge_cases()
        
        all_costs = []
        all_costs.extend(edge_cases['equal_split'].values())
        all_costs.extend(edge_cases['proportional'].values())
        
        return (min(all_costs), max(all_costs))
    
    def get_house_summary(self) -> Dict:
        """Get comprehensive house summary for UI display."""
        min_rent, max_rent = self.room_rent_range
        
        return {
            'total_rent': self.total_rent,
            'num_people': self.num_people,
            'average_rent': self.target_mean,
            'total_area': self.total_area,
            'shared_room_size': self.shared_room.size if self.shared_room else 0,
            'shared_cost_per_person': self.shared_cost_per_person,
            'individual_rent_pool': self.individual_rent_pool,
            'min_possible_rent': min_rent,
            'max_possible_rent': max_rent,
            'rent_range': max_rent - min_rent
        }
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Get room by ID."""
        for room in self.individual_rooms:
            if room.room_id == room_id:
                return room
        return None
    
    def add_room_constraint(self, room_id: str, min_acceptable: Optional[float] = None, 
                           max_acceptable: Optional[float] = None, is_active: bool = True) -> bool:
        """
        Add or update a constraint for a specific room.
        
        Args:
            room_id: ID of the room to constrain
            min_acceptable: Minimum acceptable monthly rent (optional)
            max_acceptable: Maximum acceptable monthly rent (optional)
            is_active: Whether to activate this constraint
            
        Returns:
            True if constraint was added/updated successfully
        """
        if room_id not in [room.room_id for room in self.individual_rooms]:
            return False
        
        try:
            constraint = RoomConstraint(
                room_id=room_id,
                min_acceptable=min_acceptable,
                max_acceptable=max_acceptable,
                is_active=is_active
            )
            self.room_constraints[room_id] = constraint
            return True
        except ValueError:
            return False
    
    def remove_room_constraint(self, room_id: str) -> bool:
        """Remove constraint for a specific room."""
        if room_id in self.room_constraints:
            self.room_constraints[room_id].is_active = False
            return True
        return False
    
    def get_active_constraints(self) -> List[RoomConstraint]:
        """Get all currently active constraints."""
        return [constraint for constraint in self.room_constraints.values() if constraint.is_active]
    
    def is_spread_feasible(self, spread_percent: float) -> Tuple[bool, List[str]]:
        """
        Check if a spread percentage satisfies all active constraints.
        
        Args:
            spread_percent: Spread percentage to test (0-100)
            
        Returns:
            Tuple of (is_feasible, list_of_violation_descriptions)
        """
        # Get allocation for this spread percentage
        from optimizers.quadratic_optimizer import QuadraticOptimizer
        
        optimizer = QuadraticOptimizer.from_spread_percentage(spread_percent, self)
        allocation = optimizer.optimize(self)
        
        violations = []
        
        # Check each active constraint
        for constraint in self.get_active_constraints():
            if constraint.room_id in allocation:
                rent_amount = allocation[constraint.room_id]
                if not constraint.is_satisfied_by(rent_amount):
                    violation_desc = constraint.get_violation_description(rent_amount)
                    if violation_desc:
                        violations.append(f"{constraint.room_id}: {violation_desc}")
        
        return len(violations) == 0, violations
    
    def get_feasible_spread_range(self, step_size: float = 1.0) -> Tuple[Optional[float], Optional[float], List[float]]:
        """
        Find the range of spread percentages that satisfy all active constraints.
        
        Args:
            step_size: Step size for testing spread percentages (default: 1.0%)
            
        Returns:
            Tuple of (min_feasible, max_feasible, all_feasible_spreads)
            Returns (None, None, []) if no feasible solution exists
        """
        # If no active constraints, entire range is feasible
        if not self.get_active_constraints():
            return 0.0, 100.0, list(range(0, 101, int(step_size)))
        
        feasible_spreads = []
        
        # Test spread percentages from 0 to 100
        test_spreads = [i * step_size for i in range(int(100 / step_size) + 1)]
        
        for spread_percent in test_spreads:
            is_feasible, _ = self.is_spread_feasible(spread_percent)
            if is_feasible:
                feasible_spreads.append(spread_percent)
        
        if not feasible_spreads:
            return None, None, []
        
        return min(feasible_spreads), max(feasible_spreads), feasible_spreads
    
    def get_constraint_summary(self) -> Dict[str, any]:
        """
        Get summary of all constraints and feasibility analysis.
        
        Returns:
            Dictionary with constraint analysis summary
        """
        active_constraints = self.get_active_constraints()
        min_feasible, max_feasible, feasible_spreads = self.get_feasible_spread_range()
        
        summary = {
            'total_constraints': len(self.room_constraints),
            'active_constraints': len(active_constraints),
            'constraint_details': [constraint.to_dict() for constraint in active_constraints],
            'feasible_range': {
                'min_spread': min_feasible,
                'max_spread': max_feasible,
                'total_feasible_spreads': len(feasible_spreads),
                'feasible_percentage': len(feasible_spreads) if feasible_spreads else 0
            },
            'has_feasible_solution': min_feasible is not None
        }
        
        # Add constraint impact analysis
        if min_feasible is not None and max_feasible is not None:
            constrained_range = max_feasible - min_feasible
            unconstrained_range = 100.0
            summary['constraint_impact'] = {
                'remaining_range_percent': constrained_range,
                'eliminated_range_percent': unconstrained_range - constrained_range,
                'freedom_retained': (constrained_range / unconstrained_range) * 100
            }
        
        return summary
    
    def find_constraint_conflicts(self) -> List[Dict]:
        """
        Identify constraint conflicts that make no solution possible.
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # Check if any constraints are individually impossible
        for constraint in self.get_active_constraints():
            if constraint.min_acceptable is not None and constraint.max_acceptable is not None:
                if constraint.min_acceptable > constraint.max_acceptable:
                    conflicts.append({
                        'type': 'invalid_range',
                        'room_id': constraint.room_id,
                        'description': f"{constraint.room_id} has impossible range: £{constraint.min_acceptable:.0f} - £{constraint.max_acceptable:.0f}"
                    })
        
        # Check if constraints are mathematically impossible given house total rent
        total_min_required = sum(
            constraint.min_acceptable or 0 
            for constraint in self.get_active_constraints() 
            if constraint.min_acceptable is not None
        )
        
        total_max_allowed = sum(
            constraint.max_acceptable or self.total_rent 
            for constraint in self.get_active_constraints() 
            if constraint.max_acceptable is not None
        )
        
        if total_min_required > self.total_rent:
            conflicts.append({
                'type': 'exceeds_total_rent',
                'description': f"Minimum required rents (£{total_min_required:.0f}) exceed total rent (£{self.total_rent:.0f})"
            })
        
        return conflicts
    
    def get_practical_lambda_range(self, threshold: float = 1.0, max_lambda: float = 20.0) -> float:
        """
        Calculate the practical lambda range where further increases have minimal effect.
        
        Uses sensitivity analysis to find the point of diminishing returns.
        
        Args:
            threshold: Maximum change in £ per room to consider "negligible" (default: £1)
            max_lambda: Maximum lambda to test (default: 20)
            
        Returns:
            Lambda value where further increases have minimal effect
        """
        # Avoid circular import by importing here
        from optimizers.quadratic_optimizer import QuadraticOptimizer
        
        # Cache the result to avoid expensive recalculation
        cache_key = f"practical_lambda_{threshold}_{max_lambda}_{self.total_rent}_{len(self.individual_rooms)}"
        if hasattr(self, '_lambda_cache') and cache_key in self._lambda_cache:
            return self._lambda_cache[cache_key]
        
        if not hasattr(self, '_lambda_cache'):
            self._lambda_cache = {}
        
        # Test lambda values with increasing precision
        test_lambdas = []
        
        # Coarse grid first (0, 0.5, 1.0, 1.5, ...)
        test_lambdas.extend([i * 0.5 for i in range(int(max_lambda * 2) + 1)])
        
        # Sort and ensure we have enough resolution
        test_lambdas = sorted(set(test_lambdas))
        
        previous_allocation = None
        practical_lambda = max_lambda  # Default fallback
        
        for lambda_val in test_lambdas:
            # Calculate allocation for this lambda
            optimizer = QuadraticOptimizer(lambda_val)
            current_allocation = optimizer.optimize(self)
            
            if previous_allocation is not None:
                # Calculate maximum change between this and previous allocation
                max_change = 0
                for room_id in current_allocation:
                    change = abs(current_allocation[room_id] - previous_allocation[room_id])
                    max_change = max(max_change, change)
                
                # If the maximum change is below threshold, we've found our practical limit
                if max_change < threshold:
                    practical_lambda = lambda_val
                    break
            
            previous_allocation = current_allocation.copy()
        
        # Cache the result
        self._lambda_cache[cache_key] = practical_lambda
        
        return practical_lambda
    
    def get_sensitivity_analysis(self, num_points: int = 20) -> Dict:
        """
        Get detailed sensitivity analysis data.
        
        Args:
            num_points: Number of lambda points to test
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        from optimizers.quadratic_optimizer import QuadraticOptimizer
        
        practical_max = self.get_practical_lambda_range()
        lambda_values = [i * practical_max / (num_points - 1) for i in range(num_points)]
        
        sensitivity_data = {
            'lambda_values': lambda_values,
            'allocations': {},
            'max_changes': [],
            'practical_lambda': practical_max
        }
        
        previous_allocation = None
        
        for lambda_val in lambda_values:
            optimizer = QuadraticOptimizer(lambda_val)
            allocation = optimizer.optimize(self)
            
            sensitivity_data['allocations'][lambda_val] = allocation
            
            if previous_allocation is not None:
                max_change = max(
                    abs(allocation[room_id] - previous_allocation[room_id]) 
                    for room_id in allocation
                )
                sensitivity_data['max_changes'].append(max_change)
            else:
                sensitivity_data['max_changes'].append(0)
            
            previous_allocation = allocation
        
        return sensitivity_data
    
    def get_effective_spread_range(self) -> Tuple[float, float]:
        """
        Get the effective spread percentage range (always 0% to 100%).
        
        The 100% now represents the practical maximum lambda, not a fixed value.
        
        Returns:
            Tuple of (min_spread, max_spread) - always (0.0, 100.0)
        """
        return (0.0, 100.0)
    
    def spread_to_lambda_for_house(self, spread_percent: float) -> float:
        """
        Convert spread percentage to lambda value using this house's practical range.
        
        Args:
            spread_percent: Percentage from 0-100
            
        Returns:
            Lambda value appropriate for this house
        """
        # Clamp to valid range
        spread_percent = max(0.0, min(100.0, spread_percent))
        
        # Get house-specific maximum lambda
        practical_max = self.get_practical_lambda_range()
        
        # Map 0-100% to 0-practical_max
        return (spread_percent / 100.0) * practical_max
    
    def lambda_to_spread_for_house(self, lambda_val: float) -> float:
        """
        Convert lambda value to spread percentage using this house's practical range.
        
        Args:
            lambda_val: Lambda value
            
        Returns:
            Spread percentage (0-100)
        """
        practical_max = self.get_practical_lambda_range()
        
        # Clamp lambda to practical range
        lambda_val = max(0.0, min(practical_max, lambda_val))
        
        # Map 0-practical_max to 0-100%
        return (lambda_val / practical_max) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert house to dictionary for JSON serialization."""
        return {
            'total_rent': self.total_rent,
            'individual_rooms': [room.to_dict() for room in self.individual_rooms],
            'shared_room': self.shared_room.to_dict() if self.shared_room else None,
            'room_constraints': {room_id: constraint.to_dict() for room_id, constraint in self.room_constraints.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'House':
        """Create House from dictionary for JSON deserialization."""
        individual_rooms = [Room.from_dict(room_data) for room_data in data['individual_rooms']]
        
        shared_room = None
        if data.get('shared_room'):
            shared_room = Room.from_dict(data['shared_room'])
        
        house = cls(individual_rooms, shared_room, data['total_rent'])
        
        # Restore room constraints if present
        if 'room_constraints' in data:
            for room_id, constraint_data in data['room_constraints'].items():
                constraint = RoomConstraint.from_dict(constraint_data)
                house.room_constraints[room_id] = constraint
        
        return house
    
    @classmethod
    def create_your_house(cls) -> 'House':
        """Create your specific house with manual size scores."""
        # Your house data with manual size scores (1-5)
        rooms_data = [
            {
                'id': 'Room 1',
                'size': 30.22,
                'size_score': 5.0,    # Largest room
                'noise_level': 2.5,   # Close to pub (you can adjust)
                'accessibility': 3  # Basement but close to kitchen 
            },
            {
                'id': 'Room 3', 
                'size': 19.16,
                'size_score': 4.0,    # Second largest
                'noise_level': 4.0,   # Moderate noise
                'accessibility': 4.5  # Medium accessibility
            },
            {
                'id': 'Room 4',
                'size': 16.04,
                'size_score': 3.5,    # Medium size
                'noise_level': 4.0,   # Quietest (balcony side)
                'accessibility': 4.5  # Medium accessibility
            },
            {
                'id': 'Room 5',
                'size': 15.66,
                'size_score': 3.5,    # Medium-small
                'noise_level': 4.5,   # Top floor, quiet
                'accessibility': 3.5  # Lots of stairs
            },
            {
                'id': 'Room 6',
                'size': 11.10,
                'size_score': 2.5,    # Smallest
                'noise_level': 4.5,   # Top floor, quiet
                'accessibility': 3.5  # Most stairs, furthest
            }
        ]
        
        # Create rooms
        individual_rooms = []
        for room_data in rooms_data:
            factors = DesirabilityFactors(
                size_score=room_data['size_score'],
                noise_level=room_data['noise_level'],
                accessibility=room_data['accessibility']
            )
            room = Room(
                room_id=room_data['id'],
                size=room_data['size'],
                desirability_factors=factors
            )
            individual_rooms.append(room)
        
        # Create shared room
        shared_room = Room('Room 2', 15.19, is_shared=True)
        
        return cls (individual_rooms, shared_room, 4110.0)
    
    def get_proportional_allocation(self) -> Dict[str, float]:
        """
        Get simple proportional allocation based on room size only.
        This is used as a baseline for comparison in fairness metrics.
        
        Returns:
            Dictionary of {room_id: total_monthly_cost} based on size proportion
        """
        size_proportions = self.get_size_proportions()
        
        proportional_allocation = {}
        for room_id, proportion in size_proportions.items():
            individual_cost = proportion * self.individual_rent_pool
            total_cost = individual_cost + self.shared_cost_per_person
            proportional_allocation[room_id] = total_cost
        
        return proportional_allocation
    
    def get_constrained_spread_mapping(self) -> Tuple[float, float]:
        """
        Get the actual spread range that 0-100% slider should map to.
        
        Returns:
            Tuple of (min_actual_spread, max_actual_spread)
        """
        # If no constraints, use full range
        if not self.get_active_constraints():
            return 0.0, 100.0
        
        # If constraints exist, use feasible range
        min_feasible, max_feasible, _ = self.get_feasible_spread_range()
        
        # If no feasible solution, fall back to full range
        if min_feasible is None or max_feasible is None:
            return 0.0, 100.0
        
        return min_feasible, max_feasible

    def slider_to_actual_spread(self, slider_percent: float) -> float:
        """Convert slider position (0-100%) to actual spread percentage."""
        min_actual, max_actual = self.get_constrained_spread_mapping()
        
        # Map slider 0-100% to actual min_actual-max_actual range
        slider_percent = max(0.0, min(100.0, slider_percent))
        actual_range = max_actual - min_actual
        
        return min_actual + (slider_percent / 100.0) * actual_range

    def actual_spread_to_slider(self, actual_spread: float) -> float:
        """Convert actual spread percentage to slider position (0-100%)."""
        min_actual, max_actual = self.get_constrained_spread_mapping()
        
        # Map actual spread to 0-100% slider range
        actual_spread = max(min_actual, min(max_actual, actual_spread))
        actual_range = max_actual - min_actual
        
        if actual_range == 0:
            return 0.0
        
        return ((actual_spread - min_actual) / actual_range) * 100.0