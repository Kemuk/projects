from typing import List, Dict, Optional, Union
import pandas as pd
from dataclasses import dataclass, asdict
import json


@dataclass
class DesirabilityFactors:
    """Data class to hold desirability factors for easy serialization."""
    size_score: float = 3.0
    noise_level: float = 3.0  # 1=very noisy, 5=very quiet
    floor_level: int = 1  # Floor number (0=basement, 1=ground, etc.)
    has_balcony: bool = False
    has_ensuite: bool = False
    distance_to_kitchen: str = "medium"  # close, medium, far
    natural_light: float = 3.0  # 1=poor, 5=excellent
    overall_condition: float = 3.0  # 1=poor, 5=excellent
    
    def calculate_overall_score(self) -> float:
        """Calculate overall desirability score from individual factors."""
        # Weights for different factors (should sum to 1.0)
        weights = {
            'size_score': 0.25,
            'noise_level': 0.20,
            'natural_light': 0.15,
            'overall_condition': 0.15,
            'floor_level': 0.10,  # Convert to 1-5 scale
            'has_balcony': 0.05,
            'has_ensuite': 0.05,
            'distance_to_kitchen': 0.05  # Convert to 1-5 scale
        }
        
        # Normalize some factors to 1-5 scale
        floor_score = min(5, max(1, self.floor_level + 2))  # -1=1, 0=2, 1=3, 2=4, 3+=5
        
        distance_scores = {"close": 5, "medium": 3, "far": 1}
        kitchen_score = distance_scores.get(self.distance_to_kitchen, 3)
        
        # Calculate weighted score
        score = (
            self.size_score * weights['size_score'] +
            self.noise_level * weights['noise_level'] +
            self.natural_light * weights['natural_light'] +
            self.overall_condition * weights['overall_condition'] +
            floor_score * weights['floor_level'] +
            (5 if self.has_balcony else 3) * weights['has_balcony'] +
            (5 if self.has_ensuite else 3) * weights['has_ensuite'] +
            kitchen_score * weights['distance_to_kitchen']
        )
        
        return round(score, 2)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DesirabilityFactors':
        """Create from dictionary for JSON deserialization."""
        return cls(**data)


class Room:
    """Represents a single room with its properties and desirability factors."""
    
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
        self.proportional_cost = 0.0
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
    
    def to_dict(self) -> Dict:
        """Convert room to dictionary representation for JSON serialization."""
        return {
            'room_id': self.room_id,
            'size': self.size,
            'is_shared': self.is_shared,
            'desirability_factors': self.desirability_factors.to_dict(),
            'desirability_score': self.desirability_score,
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
    
    def __repr__(self):
        return f"Room(id='{self.room_id}', size={self.size}, desirability={self.desirability_score})"

class House:
    """
    Represents a house containing multiple rooms and handles house-level calculations.
    Improved for Streamlit compatibility and robustness.
    """
    
    def __init__(self, 
                 individual_rooms: List[Room], 
                 shared_room: Optional[Room] = None, 
                 total_rent: float = 0.0):
        """
        Initialize a House object.
        
        Args:
            individual_rooms: List of Room objects for individual bedrooms
            shared_room: Room object for the shared living space (optional)
            total_rent: Total monthly rent for the house
        """
        # Validation
        if not individual_rooms:
            raise ValueError("House must have at least one individual room")
        if total_rent <= 0:
            raise ValueError("Total rent must be positive")
        
        # Validate unique room IDs
        room_ids = [room.room_id for room in individual_rooms]
        if shared_room:
            room_ids.append(shared_room.room_id)
        if len(room_ids) != len(set(room_ids)):
            raise ValueError("All room IDs must be unique")
        
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
        
        self._calculate_proportional_costs()
    
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
    
    def _calculate_proportional_costs(self):
        """Calculate proportional allocation based on room sizes."""
        total_individual_area = sum(room.size for room in self.individual_rooms)
        
        if total_individual_area == 0:
            raise ValueError("Total individual room area cannot be zero")
        
        for room in self.individual_rooms:
            individual_cost = (room.size / total_individual_area) * self.individual_rent_pool
            room.set_costs(individual_cost, self.shared_cost_per_person)
    
    def add_room(self, room: Room):
        """Add a new individual room to the house."""
        if room.room_id in [r.room_id for r in self.individual_rooms]:
            raise ValueError(f"Room ID '{room.room_id}' already exists")
        
        self.individual_rooms.append(room)
        self.num_people = len(self.individual_rooms)
        self.target_mean = self.total_rent / self.num_people
        self.total_area = self._calculate_total_area()
        
        # Recalculate costs
        if self.shared_room:
            self._calculate_shared_costs()
        self._calculate_proportional_costs()
    
    def remove_room(self, room_id: str):
        """Remove a room from the house."""
        if len(self.individual_rooms) <= 1:
            raise ValueError("Cannot remove room - house must have at least one individual room")
        
        room_to_remove = None
        for room in self.individual_rooms:
            if room.room_id == room_id:
                room_to_remove = room
                break
        
        if not room_to_remove:
            raise ValueError(f"Room '{room_id}' not found")
        
        self.individual_rooms.remove(room_to_remove)
        self.num_people = len(self.individual_rooms)
        self.target_mean = self.total_rent / self.num_people
        self.total_area = self._calculate_total_area()
        
        # Recalculate costs
        if self.shared_room:
            self._calculate_shared_costs()
        self._calculate_proportional_costs()
    
    def update_total_rent(self, new_rent: float):
        """Update the total rent and recalculate all costs."""
        if new_rent <= 0:
            raise ValueError("Total rent must be positive")
        
        self.total_rent = new_rent
        self.target_mean = new_rent / self.num_people
        
        # Recalculate costs
        if self.shared_room:
            self._calculate_shared_costs()
        self._calculate_proportional_costs()
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Get a room by its ID."""
        for room in self.individual_rooms:
            if room.room_id == room_id:
                return room
        if self.shared_room and self.shared_room.room_id == room_id:
            return self.shared_room
        return None
    
    def get_all_rooms(self) -> List[Room]:
        """Get all rooms (individual + shared)."""
        rooms = self.individual_rooms.copy()
        if self.shared_room:
            rooms.append(self.shared_room)
        return rooms
    
    def get_proportional_allocation(self) -> Dict[str, float]:
        """Get proportional allocation as a dictionary."""
        return {room.room_id: room.total_cost for room in self.individual_rooms}
    
    def apply_allocation(self, allocation: Dict[str, float]):
        """Apply a new cost allocation to the rooms."""
        if not self.validate_allocation(allocation):
            raise ValueError("Allocation does not sum to total rent")
        
        for room in self.individual_rooms:
            if room.room_id in allocation:
                new_total_cost = allocation[room.room_id]
                new_individual_cost = new_total_cost - self.shared_cost_per_person
                if new_individual_cost < 0:
                    raise ValueError(f"Individual cost for {room.room_id} cannot be negative")
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
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with all room information."""
        data = []
        
        # Add individual rooms
        for room in self.individual_rooms:
            data.append({
                'Room_ID': room.room_id,
                'Type': 'Individual',
                'Size_sqm': room.size,
                'Desirability_Score': round(room.desirability_score, 2),
                'Individual_Cost': round(room.individual_cost, 2),
                'Shared_Cost_Portion': round(room.shared_cost_portion, 2),
                'Total_Monthly_Cost': round(room.total_cost, 2),
                'Cost_Per_sqm': round(room.total_cost / room.size, 2),
                'Deviation_from_Mean': round(room.total_cost - self.target_mean, 2)
            })
        
        # Add shared room if it exists
        if self.shared_room:
            data.append({
                'Room_ID': self.shared_room.room_id,
                'Type': 'Shared',
                'Size_sqm': self.shared_room.size,
                'Desirability_Score': None,
                'Individual_Cost': 0,
                'Shared_Cost_Portion': round(self.shared_room_total_cost, 2),
                'Total_Monthly_Cost': round(self.shared_room_total_cost, 2),
                'Cost_Per_sqm': round(self.shared_room_total_cost / self.shared_room.size, 2),
                'Deviation_from_Mean': None
            })
        
        return pd.DataFrame(data)
    
    def get_streamlit_room_data(self) -> Dict:
        """Get room data in format suitable for Streamlit widgets."""
        rooms_data = {}
        for i, room in enumerate(self.individual_rooms):
            rooms_data[f"room_{i}"] = {
                'id': room.room_id,
                'size': room.size,
                'desirability_factors': room.desirability_factors.to_dict()
            }
        
        if self.shared_room:
            rooms_data['shared_room'] = {
                'id': self.shared_room.room_id,
                'size': self.shared_room.size
            }
        
        return rooms_data
    
    def to_dict(self) -> Dict:
        """Convert house to dictionary for JSON serialization."""
        data = {
            'total_rent': self.total_rent,
            'individual_rooms': [room.to_dict() for room in self.individual_rooms],
            'shared_room': self.shared_room.to_dict() if self.shared_room else None
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'House':
        """Create House from dictionary for JSON deserialization."""
        # Recreate individual rooms
        individual_rooms = [Room.from_dict(room_data) for room_data in data['individual_rooms']]
        
        # Recreate shared room if it exists
        shared_room = None
        if data.get('shared_room'):
            shared_room = Room.from_dict(data['shared_room'])
        
        return cls(individual_rooms, shared_room, data['total_rent'])
    
    @classmethod
    def create_from_streamlit_data(cls, 
                                  total_rent: float,
                                  rooms_data: List[Dict],
                                  shared_room_data: Optional[Dict] = None) -> 'House':
        """
        Create House from Streamlit form data.
        
        Args:
            total_rent: Total monthly rent
            rooms_data: List of dicts with room information
            shared_room_data: Dict with shared room info (optional)
        """
        # Create individual rooms
        individual_rooms = []
        for room_data in rooms_data:
            # Create desirability factors
            desirability_factors = DesirabilityFactors(
                size_score=room_data.get('size_score', 3.0),
                noise_level=room_data.get('noise_level', 3.0),
                floor_level=room_data.get('floor_level', 1),
                has_balcony=room_data.get('has_balcony', False),
                has_ensuite=room_data.get('has_ensuite', False),
                distance_to_kitchen=room_data.get('distance_to_kitchen', 'medium'),
                natural_light=room_data.get('natural_light', 3.0),
                overall_condition=room_data.get('overall_condition', 3.0)
            )
            
            room = Room(
                room_id=room_data['id'],
                size=room_data['size'],
                desirability_factors=desirability_factors
            )
            individual_rooms.append(room)
        
        # Create shared room if provided
        shared_room = None
        if shared_room_data:
            shared_room = Room(
                room_id=shared_room_data['id'],
                size=shared_room_data['size'],
                is_shared=True
            )
        
        return cls(individual_rooms, shared_room, total_rent)
    
    def print_summary(self):
        """Print a summary of the house information."""
        print("🏠 HOUSE SUMMARY")
        print("=" * 50)
        print(f"Total rent: £{self.total_rent}")
        print(f"Number of people: {self.num_people}")
        print(f"Target mean per person: £{self.target_mean:.2f}")
        print(f"Total area: {self.total_area:.2f} sqm")
        
        if self.shared_room:
            print(f"\n🛏️ SHARED ROOM")
            print("-" * 20)
            print(f"{self.shared_room.room_id}: {self.shared_room.size} sqm")
            print(f"Total cost: £{self.shared_room_total_cost:.2f}")
            print(f"Cost per person: £{self.shared_cost_per_person:.2f}")
        
        print(f"\n🚪 INDIVIDUAL ROOMS")
        print("-" * 30)
        for room in self.individual_rooms:
            print(f"{room.room_id}: {room.size} sqm")
            print(f"  Desirability: {room.desirability_score:.2f}/5")
            print(f"  Individual cost: £{room.individual_cost:.2f}")
            print(f"  Total cost: £{room.total_cost:.2f}")
    
    def __repr__(self):
        return f"House(people={self.num_people}, total_rent=£{self.total_rent}, rooms={len(self.individual_rooms)})"