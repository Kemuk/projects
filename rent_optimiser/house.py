from typing import List, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import json


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
    
    def to_dict(self) -> Dict:
        """Convert house to dictionary for JSON serialization."""
        return {
            'total_rent': self.total_rent,
            'individual_rooms': [room.to_dict() for room in self.individual_rooms],
            'shared_room': self.shared_room.to_dict() if self.shared_room else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'House':
        """Create House from dictionary for JSON deserialization."""
        individual_rooms = [Room.from_dict(room_data) for room_data in data['individual_rooms']]
        
        shared_room = None
        if data.get('shared_room'):
            shared_room = Room.from_dict(data['shared_room'])
        
        return cls(individual_rooms, shared_room, data['total_rent'])
    
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
    