from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass


class Room:
    """
    Represents a single room with its properties and desirability factors.
    """
    
    def __init__(self, 
                 room_id: str, 
                 size: float, 
                 desirability_score: Optional[float] = None,
                 is_shared: bool = False):
        """
        Initialize a Room object.
        
        Args:
            room_id: Unique identifier for the room (e.g., "Room 1")
            size: Size of the room in square meters
            desirability_score: Overall desirability score (0-5), calculated if None
            is_shared: Whether this room is shared among all housemates
        """
        self.room_id = room_id
        self.size = size
        self.desirability_score = desirability_score
        self.is_shared = is_shared
        self.desirability_factors = {}
        
        # Cost-related attributes (set by House object)
        self.proportional_cost = 0.0
        self.individual_cost = 0.0
        self.shared_cost_portion = 0.0
        self.total_cost = 0.0
    
    def calculate_desirability_factors(self) -> Dict[str, float]:
        """
        Calculate desirability factors for this room based on size, noise, kitchen and bathroom access.
        
        Returns:
            Dictionary of factor scores (out of 5)
        """
        # Size Score (1-5 scale)
        size_scores = {
            'Room 1': 5,  # Largest room by far
            'Room 3': 4,  # Second largest
            'Room 4': 3,  # Medium size
            'Room 5': 3,  # Medium-small
            'Room 6': 2.5   # Smallest room
        }
        
        # Noise Level Score (1-5 scale, higher = quieter)
        noise_scores = {
            'Room 1': 2,  # Basement close to street
            'Room 3': 4,  # 2nd floor, away from pub
            'Room 4': 5,  # Balcony side, furthest from pub
            'Room 5': 5,  # Top floor, furthest from pub
            'Room 6': 5   # Top floor, away from pub
        }
        
        # Kitchen Convenience Score (1-5 scale)
        kitchen_scores = {
            'Room 1': 4,  # Basement, but one floor
            'Room 3': 4,  # One floor up, reasonable
            'Room 4': 4,  # One floor up from kitchen
            'Room 5': 3,  # Two floors up, lots of stairs
            'Room 6': 3,   # Furthest from kitchen, most stairs
        }
        
        # Bathroom Convenience Score (1-5 scale)
        bathroom_scores = {
            'Room 1': 3,  # Own bathroom - no sharing/waiting, but basement location
            'Room 3': 5,  # Main floor bathroom, most convenient location
            'Room 4': 5,  # Main floor bathroom, most convenient location
            'Room 5': 5,  # Top floor - less convenient when in main house areas
            'Room 6': 5   # Top floor - less convenient when in main house areas
        }
        
        # Calculate individual factors
        factors = {
            'size': size_scores.get(self.room_id, 3),
            'noise': noise_scores.get(self.room_id, 3),
            'kitchen_convenience': kitchen_scores.get(self.room_id, 3),
            'bathroom_convenience': bathroom_scores.get(self.room_id, 3)
        }
        
        # Calculate overall desirability score (weighted average)
        weights = {
            'size': 0.45,
            'noise': 0.35,
            'kitchen_convenience': 0.1,
            'bathroom_convenience': 0.1
        }
        
        overall_score = sum(factors[factor] * weight for factor, weight in weights.items())
        factors['overall_desirability'] = round(overall_score, 2)
        
        self.desirability_factors = factors
        self.desirability_score = factors['overall_desirability']
        
        return factors
    
    def set_costs(self, individual_cost: float, shared_cost_portion: float):
        """Set the costs for this room."""
        self.individual_cost = individual_cost
        self.shared_cost_portion = shared_cost_portion
        self.total_cost = individual_cost + shared_cost_portion
    
    def to_dict(self) -> Dict:
        """Convert room to dictionary representation."""
        return {
            'room_id': self.room_id,
            'size': self.size,
            'desirability_score': self.desirability_score,
            'is_shared': self.is_shared,
            'individual_cost': self.individual_cost,
            'shared_cost_portion': self.shared_cost_portion,
            'total_cost': self.total_cost,
            'desirability_factors': self.desirability_factors
        }
    
    def __repr__(self):
        return f"Room(id='{self.room_id}', size={self.size}, desirability={self.desirability_score})"


class House:
    """
    Represents a house containing multiple rooms and handles house-level calculations.
    """
    
    def __init__(self, 
                 individual_rooms: List[Room], 
                 shared_room: Room, 
                 total_rent: float):
        """
        Initialize a House object.
        
        Args:
            individual_rooms: List of Room objects for individual bedrooms
            shared_room: Room object for the shared living space
            total_rent: Total monthly rent for the house
        """
        self.individual_rooms = individual_rooms
        self.shared_room = shared_room
        self.total_rent = total_rent
        
        # Calculated properties
        self.num_people = len(individual_rooms)
        self.total_area = sum(room.size for room in individual_rooms) + shared_room.size
        self.target_mean = total_rent / self.num_people
        
        # Calculate shared room costs
        self._calculate_shared_costs()
        
        # Calculate proportional costs for each room
        self._calculate_proportional_costs()
    
    def _calculate_shared_costs(self):
        """Calculate costs related to the shared room."""
        self.shared_room_total_cost = (self.shared_room.size / self.total_area) * self.total_rent
        self.shared_cost_per_person = self.shared_room_total_cost / self.num_people
        self.individual_rent_pool = self.total_rent - self.shared_room_total_cost
        
        # Set shared room costs
        self.shared_room.set_costs(
            individual_cost=0,  # Shared room has no individual cost
            shared_cost_portion=self.shared_room_total_cost
        )
    
    def _calculate_proportional_costs(self):
        """Calculate proportional allocation based on room sizes."""
        total_individual_area = sum(room.size for room in self.individual_rooms)
        
        for room in self.individual_rooms:
            individual_cost = (room.size / total_individual_area) * self.individual_rent_pool
            room.set_costs(individual_cost, self.shared_cost_per_person)
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Get a room by its ID."""
        for room in self.individual_rooms:
            if room.room_id == room_id:
                return room
        if self.shared_room.room_id == room_id:
            return self.shared_room
        return None
    
    def get_all_rooms(self) -> List[Room]:
        """Get all rooms (individual + shared)."""
        return self.individual_rooms + [self.shared_room]
    
    def calculate_desirability_for_all_rooms(self):
        """Calculate desirability factors for all individual rooms."""
        for room in self.individual_rooms:
            room.calculate_desirability_factors()
    
    def get_proportional_allocation(self) -> Dict[str, float]:
        """Get proportional allocation as a dictionary."""
        return {room.room_id: room.total_cost for room in self.individual_rooms}
    
    def get_individual_costs(self) -> Dict[str, float]:
        """Get individual room costs (excluding shared portion)."""
        return {room.room_id: room.individual_cost for room in self.individual_rooms}
    
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
        total = sum(allocation.values())
        return abs(total - self.total_rent) < 0.01
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with all room information."""
        data = []
        
        # Add individual rooms
        for room in self.individual_rooms:
            data.append({
                'Room ID': room.room_id,
                'Type': 'Individual',
                'Size (sqm)': room.size,
                'Desirability Score': room.desirability_score or 'N/A',
                'Individual Cost (£)': round(room.individual_cost, 2),
                'Shared Cost Portion (£)': round(room.shared_cost_portion, 2),
                'Total Cost (£)': round(room.total_cost, 2),
                'Deviation from Mean (£)': round(room.total_cost - self.target_mean, 2)
            })
        
        # Add shared room
        data.append({
            'Room ID': self.shared_room.room_id,
            'Type': 'Shared',
            'Size (sqm)': self.shared_room.size,
            'Desirability Score': 'N/A',
            'Individual Cost (£)': 0,
            'Shared Cost Portion (£)': round(self.shared_room_total_cost, 2),
            'Total Cost (£)': round(self.shared_room_total_cost, 2),
            'Deviation from Mean (£)': 'N/A'
        })
        
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print a summary of the house information."""
        print("🏠 HOUSE SUMMARY")
        print("=" * 50)
        print(f"Total rent: £{self.total_rent}")
        print(f"Number of people: {self.num_people}")
        print(f"Target mean per person: £{self.target_mean:.2f}")
        print(f"Total area: {self.total_area:.2f} sqm")
        
        print(f"\n🛏️ SHARED ROOM")
        print("-" * 20)
        print(f"{self.shared_room.room_id}: {self.shared_room.size} sqm")
        print(f"Total cost: £{self.shared_room_total_cost:.2f}")
        print(f"Cost per person: £{self.shared_cost_per_person:.2f}")
        
        print(f"\n🚪 INDIVIDUAL ROOMS")
        print("-" * 30)
        for room in self.individual_rooms:
            desirability = room.desirability_score or 'N/A'
            print(f"{room.room_id}: {room.size} sqm")
            print(f"  Desirability: {desirability}/5")
            print(f"  Individual cost: £{room.individual_cost:.2f}")
            print(f"  Total cost: £{room.total_cost:.2f}")
    
    def print_desirability_analysis(self):
        """Print detailed desirability analysis."""
        print("\n🎯 DESIRABILITY ANALYSIS")
        print("=" * 70)
        
        for room in self.individual_rooms:
            if not room.desirability_factors:
                continue
                
            print(f"\n📍 {room.room_id} ({room.size} sqm) - Overall: {room.desirability_score}/5")
            print("-" * 50)
            
            factor_descriptions = {
                'size': 'Room Size',
                'noise': 'Noise Level (Higher = Quieter)',
                'kitchen_convenience': 'Kitchen Convenience',
                'bathroom_convenience': 'Bathroom Convenience'
            }
            
            for factor, score in room.desirability_factors.items():
                if factor != 'overall_desirability':
                    description = factor_descriptions.get(factor, factor)
                    print(f"  {description:<30}: {score}/5")
    
    @classmethod
    def create_from_data(cls, 
                        room_data: Dict[str, float], 
                        total_rent: float,
                        shared_room_id: str,
                        shared_room_size: float) -> 'House':
        """
        Create a House object from room data dictionary.
        
        Args:
            room_data: Dictionary of {room_id: size} for individual rooms
            total_rent: Total monthly rent
            shared_room_id: ID of the shared room
            shared_room_size: Size of the shared room
            
        Returns:
            House object
        """
        # Create individual room objects
        individual_rooms = [Room(room_id, size) for room_id, size in room_data.items()]
        
        # Create shared room object
        shared_room = Room(shared_room_id, shared_room_size, is_shared=True)
        
        return cls(individual_rooms, shared_room, total_rent)
    
    def __repr__(self):
        return f"House(people={self.num_people}, total_rent=£{self.total_rent}, rooms={len(self.individual_rooms)})"