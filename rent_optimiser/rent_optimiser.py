import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class RentOptimizer:
    """
    A comprehensive tool for optimizing rent allocation among housemates.
    
    Balances equality (everyone pays similar amounts) with fairness 
    (larger rooms pay proportionally more) and desirability factors.
    """
    
    def __init__(self, room_sizes: Dict[str, float], total_rent: float, 
                 shared_room_size: float, target_mean: float = None, 
                 desirability_scores: Dict[str, float] = None):
        """
        Initialize the rent optimizer.
        
        Args:
            room_sizes: Dictionary of {room_id: size_in_sqm}
            total_rent: Total monthly rent for the house
            shared_room_size: Size of shared room (split equally)
            target_mean: Target mean rent per person (calculated if None)
            desirability_scores: Dictionary of {room_id: desirability_score} (0-10 scale)
        """
        self.room_sizes = room_sizes
        self.total_rent = total_rent
        self.shared_room_size = shared_room_size
        self.num_people = len(room_sizes)
        self.desirability_scores = desirability_scores or {}
        
        # Calculate shared room cost
        total_area = sum(room_sizes.values()) + shared_room_size
        self.shared_room_cost = (shared_room_size / total_area) * total_rent
        self.shared_cost_per_person = self.shared_room_cost / self.num_people
        
        # Remaining rent for individual rooms
        self.individual_rent = total_rent - self.shared_room_cost
        
        # Target mean (total rent / number of people)
        self.target_mean = target_mean or (total_rent / self.num_people)
        
        # Calculate baseline proportional allocation
        self.proportional_costs = self._calculate_proportional()
        
    def set_desirability_scores(self, desirability_scores: Dict[str, float]):
        """Set or update desirability scores for rooms."""
        self.desirability_scores = desirability_scores
        
    def calculate_desirability_factors(self, pub_proximity: Dict[str, str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate desirability factors for each room based on floor plan analysis.
        
        Args:
            pub_proximity: Dictionary of {room_id: 'close'|'medium'|'far'} from pub
            
        Returns:
            Dictionary with detailed desirability breakdown per room
        """
        # Default pub proximity if not provided
        if pub_proximity is None:
            pub_proximity = {
                'Room 1': 'close',    # Ground floor (basement) - likely closest to pub
                'Room 2': 'medium',   # 1st floor
                'Room 3': 'medium',   # 2nd floor  
                'Room 4': 'far',      # 2nd floor with balcony - might be away from pub
                'Room 5': 'far',      # 3rd floor - highest, furthest from pub
                'Room 6': 'far'       # 3rd floor
            }
        
        factors = {}
        
        for room_id in self.room_sizes.keys():
            room_factors = {}
            
            # 1. Floor Level Score (higher = better for most people)
            floor_scores = {
                'Room 1': 3,  # Basement/Ground floor - less desirable
                'Room 2': 7,  # 1st floor - convenient to kitchen
                'Room 3': 8,  # 2nd floor - good height
                'Room 4': 9,  # 2nd floor with balcony - very desirable
                'Room 5': 6,  # 3rd floor - lots of stairs
                'Room 6': 5   # 3rd floor - lots of stairs, smaller
            }
            room_factors['floor_level'] = floor_scores.get(room_id, 5)
            
            # 2. Natural Light Score (based on floor plan analysis)
            light_scores = {
                'Room 1': 4,  # Basement - limited natural light
                'Room 2': 6,  # Ground level windows
                'Room 3': 7,  # Good windows on 2nd floor
                'Room 4': 10, # Balcony + large windows - excellent light
                'Room 5': 8,  # Top floor - good light
                'Room 6': 7   # Top floor but smaller windows
            }
            room_factors['natural_light'] = light_scores.get(room_id, 5)
            
            # 3. Noise Level (pub closes midnight, garden 10pm)
            noise_penalties = {
                'close': -3,   # Basement likely closest to pub noise
                'medium': -1,  # Mid-level floors
                'far': 0       # Top floors, away from pub
            }
            room_factors['noise_level'] = 5 + noise_penalties.get(pub_proximity.get(room_id, 'medium'), -1)
            
            # 4. Privacy Score (distance from common areas)
            privacy_scores = {
                'Room 1': 9,  # Basement - very private
                'Room 2': 4,  # Same floor as kitchen - less private
                'Room 3': 7,  # Separate floor from kitchen
                'Room 4': 8,  # Good separation + balcony
                'Room 5': 8,  # Top floor - private
                'Room 6': 7   # Top floor but near Room 5
            }
            room_factors['privacy'] = privacy_scores.get(room_id, 5)
            
            # 5. Convenience Score (proximity to kitchen/bathroom)
            convenience_scores = {
                'Room 1': 3,  # Basement - far from kitchen, stairs
                'Room 2': 10, # Same floor as kitchen - very convenient
                'Room 3': 6,  # One floor up from kitchen
                'Room 4': 6,  # One floor up from kitchen
                'Room 5': 4,  # Two floors up - lots of stairs
                'Room 6': 4   # Two floors up - lots of stairs
            }
            room_factors['convenience'] = convenience_scores.get(room_id, 5)
            
            # 6. Special Features & Balcony Access
            special_features = {
                'Room 1': 2,  # Basement ceiling, less desirable
                'Room 2': 5,  # Standard room
                'Room 3': 6,  # Standard room, good size
                'Room 4': 7,  # Adjacent to communal balcony - easy access but not exclusive
                'Room 5': 7,  # Top floor views
                'Room 6': 4   # Smaller, but top floor
            }
            room_factors['special_features'] = special_features.get(room_id, 5)
            
            # 7. Balcony Access (since balcony is communal)
            balcony_access = {
                'Room 1': 3,  # Basement - furthest from communal balcony
                'Room 2': 4,  # Different floor from balcony
                'Room 3': 6,  # Same floor as balcony but not adjacent
                'Room 4': 9,  # Adjacent to communal balcony - easiest access
                'Room 5': 4,  # Different floor from balcony
                'Room 6': 4   # Different floor from balcony
            }
            room_factors['balcony_access'] = balcony_access.get(room_id, 5)
            
            # 8. Calculate overall desirability score (weighted average)
            weights = {
                'floor_level': 0.15,
                'natural_light': 0.18,
                'noise_level': 0.20,
                'privacy': 0.15,
                'convenience': 0.15,
                'special_features': 0.12,
                'balcony_access': 0.05  # Small weight since balcony is communal
            }
            
            overall_score = sum(room_factors[factor] * weight 
                              for factor, weight in weights.items())
            room_factors['overall_desirability'] = round(overall_score, 2)
            
            factors[room_id] = room_factors
            
        return factors
        
    def _calculate_proportional(self) -> Dict[str, float]:
        """Calculate proportional allocation based on room sizes."""
        total_individual_area = sum(self.room_sizes.values())
        proportional_costs = {}
        
        for room_id, size in self.room_sizes.items():
            room_cost = (size / total_individual_area) * self.individual_rent
            total_cost = room_cost + self.shared_cost_per_person
            proportional_costs[room_id] = total_cost
            
        return proportional_costs
    
    def _calculate_size_desirability_weighted(self, desirability_weight: float = 0.3) -> Dict[str, float]:
        """
        Calculate allocation based on both room size and desirability.
        
        Args:
            desirability_weight: Weight for desirability (0-1). Higher = more weight on desirability
        """
        if not self.desirability_scores:
            return self.proportional_costs
            
        total_individual_area = sum(self.room_sizes.values())
        
        # Normalize desirability scores to 0-1 range
        max_desirability = max(self.desirability_scores.values())
        min_desirability = min(self.desirability_scores.values())
        desirability_range = max_desirability - min_desirability or 1
        
        weighted_costs = {}
        total_weighted_value = 0
        
        # Calculate weighted value for each room
        for room_id, size in self.room_sizes.items():
            desirability = self.desirability_scores.get(room_id, 5)
            normalized_desirability = (desirability - min_desirability) / desirability_range
            
            # Combine size and desirability
            size_weight = 1 - desirability_weight
            weighted_value = (size_weight * size + 
                            desirability_weight * normalized_desirability * (size * 0.5))
            total_weighted_value += weighted_value
        
        # Allocate costs based on weighted values
        for room_id, size in self.room_sizes.items():
            desirability = self.desirability_scores.get(room_id, 5)
            normalized_desirability = (desirability - min_desirability) / desirability_range
            
            size_weight = 1 - desirability_weight
            weighted_value = (size_weight * size + 
                            desirability_weight * normalized_desirability * (size * 0.5))
            
            room_cost = (weighted_value / total_weighted_value) * self.individual_rent
            total_cost = room_cost + self.shared_cost_per_person
            weighted_costs[room_id] = total_cost
            
        return weighted_costs
    
    def equal_payment(self) -> Dict[str, float]:
        """Everyone pays exactly the target mean (total including shared room)."""
        return {room_id: self.target_mean for room_id in self.room_sizes.keys()}
    
    def quadratic_optimization(self, lambda_param: float = 1.0) -> Dict[str, float]:
        """
        Quadratic programming optimization.
        
        Minimizes: Σ(cost_i - target_mean)² + λ * Σ(cost_i - proportional_i)²
        
        Args:
            lambda_param: Weight for proportional deviation penalty
                         λ = 0: Pure equality (everyone pays target_mean)
                         λ = ∞: Pure proportional allocation
                         λ = 1: Balanced approach
        """
        room_ids = list(self.room_sizes.keys())
        proportional_values = [self.proportional_costs[room_id] for room_id in room_ids]
        
        # For quadratic objective with linear constraint, analytical solution exists:
        # cost_i = (target_mean + λ * proportional_i) / (1 + λ)
        preliminary_costs = [
            (self.target_mean + lambda_param * prop_cost) / (1 + lambda_param)
            for prop_cost in proportional_values
        ]
        
        # Adjust to ensure total equals total_rent
        preliminary_total = sum(preliminary_costs)
        adjustment = (self.total_rent - preliminary_total) / self.num_people
        
        optimized_costs = [cost + adjustment for cost in preliminary_costs]
        
        return dict(zip(room_ids, optimized_costs))
    
    def get_individual_room_costs(self, total_costs: Dict[str, float]) -> Dict[str, float]:
        """Convert total costs back to individual room costs (excluding shared room)."""
        return {room_id: total_cost - self.shared_cost_per_person 
                for room_id, total_cost in total_costs.items()}
    
    def progressive_cap(self, max_above_mean: float = 100, 
                       max_below_mean: float = 150) -> Dict[str, float]:
        """
        Progressive cap optimization.
        
        Limits how much anyone can deviate from the target mean, then
        redistributes to maintain total rent.
        
        Args:
            max_above_mean: Maximum amount above target mean
            max_below_mean: Maximum amount below target mean
        """
        # Apply caps to proportional costs
        capped_costs = {}
        for room_id, prop_cost in self.proportional_costs.items():
            max_cost = self.target_mean + max_above_mean
            min_cost = self.target_mean - max_below_mean
            capped_costs[room_id] = max(min_cost, min(max_cost, prop_cost))
        
        # Redistribute to maintain total
        capped_total = sum(capped_costs.values())
        redistribution = (self.total_rent - capped_total) / self.num_people
        
        return {room_id: cost + redistribution 
                for room_id, cost in capped_costs.items()}
    
    def minimize_max_deviation(self) -> Dict[str, float]:
        """
        Minimize the maximum deviation from target mean (Rawlsian approach).
        Uses scipy.optimize to find the solution.
        """
        room_ids = list(self.room_sizes.keys())
        n = len(room_ids)
        
        def objective(costs):
            # Minimize maximum absolute deviation from target mean
            deviations = [abs(cost - self.target_mean) for cost in costs]
            return max(deviations)
        
        def constraint(costs):
            # Total must equal total_rent
            return sum(costs) - self.total_rent
        
        # Initial guess: proportional allocation
        initial_guess = [self.proportional_costs[room_id] for room_id in room_ids]
        
        # Constraints
        constraints = {'type': 'eq', 'fun': constraint}
        
        # Bounds: costs must be positive and reasonable
        bounds = [(100, 2000) for _ in range(n)]
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                         constraints=constraints, bounds=bounds)
        
        if result.success:
            return dict(zip(room_ids, result.x))
        else:
            print("Optimization failed, returning proportional allocation")
            return self.proportional_costs
    
    def compare_methods(self, include_desirability: bool = False) -> pd.DataFrame:
        """Compare all optimization methods in a DataFrame."""
        methods = {
            'Proportional': self.proportional_costs,
            'Equal Payment': self.equal_payment(),
            'Quadratic (λ=0.25)': self.quadratic_optimization(0.25),
            'Quadratic (λ=1.0)': self.quadratic_optimization(1.0),
            'Progressive Cap': self.progressive_cap(),
            'Min Max Deviation': self.minimize_max_deviation()
        }
        
        # Add desirability-adjusted methods if desirability scores are available
        if include_desirability and self.desirability_scores:
            methods['Size + Desirability (30%)'] = self._calculate_size_desirability_weighted(0.3)
            methods['Size + Desirability (50%)'] = self._calculate_size_desirability_weighted(0.5)
        
        # Create comparison DataFrame
        comparison_data = []
        for method_name, costs in methods.items():
            for room_id, total_cost in costs.items():
                room_size = self.room_sizes[room_id]
                individual_room_cost = total_cost - self.shared_cost_per_person
                deviation_from_mean = total_cost - self.target_mean
                deviation_from_prop = total_cost - self.proportional_costs[room_id]
                desirability = self.desirability_scores.get(room_id, 'N/A')
                
                comparison_data.append({
                    'Method': method_name,
                    'Room': room_id,
                    'Size (sqm)': room_size,
                    'Desirability': desirability,
                    'Individual Room Cost (£)': round(individual_room_cost, 2),
                    'Shared Room Cost (£)': round(self.shared_cost_per_person, 2),
                    'Total Cost (£)': round(total_cost, 2),
                    'Deviation from Mean (£)': round(deviation_from_mean, 2),
                    'Change from Proportional (£)': round(deviation_from_prop, 2)
                })
        
        return pd.DataFrame(comparison_data)
    
    def print_desirability_analysis(self, pub_proximity: Dict[str, str] = None):
        """Print detailed desirability analysis for all rooms."""
        factors = self.calculate_desirability_factors(pub_proximity)
        
        print("\n🏠 ROOM DESIRABILITY ANALYSIS")
        print("=" * 70)
        print("Based on floor plan, pub proximity, and room characteristics")
        print()
        
        for room_id in sorted(self.room_sizes.keys()):
            room_factors = factors[room_id]
            size = self.room_sizes[room_id]
            
            print(f"📍 {room_id} ({size} sqm) - Overall Score: {room_factors['overall_desirability']}/10")
            print("-" * 50)
            
            factor_descriptions = {
                'floor_level': 'Floor Level',
                'natural_light': 'Natural Light',
                'noise_level': 'Noise Level (pub impact)',
                'privacy': 'Privacy',
                'convenience': 'Kitchen/Bathroom Access',
                'special_features': 'Special Features',
                'balcony_access': 'Communal Balcony Access'
            }
            
            for factor, score in room_factors.items():
                if factor != 'overall_desirability':
                    description = factor_descriptions.get(factor, factor)
                    print(f"  {description:<25}: {score}/10")
            print()
        
        # Update desirability scores
        overall_scores = {room_id: factors[room_id]['overall_desirability'] 
                         for room_id in self.room_sizes.keys()}
        self.set_desirability_scores(overall_scores)
        
        return factors
    
    def calculate_fairness_metrics(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Calculate various fairness metrics for a cost allocation."""
        cost_values = list(costs.values())
        
        # Deviation from target mean
        mean_deviations = [abs(cost - self.target_mean) for cost in cost_values]
        avg_deviation_from_mean = np.mean(mean_deviations)
        max_deviation_from_mean = max(mean_deviations)
        
        # Deviation from proportional
        prop_deviations = [abs(costs[room_id] - self.proportional_costs[room_id]) 
                          for room_id in costs.keys()]
        avg_deviation_from_prop = np.mean(prop_deviations)
        
        # Variance and standard deviation
        variance = np.var(cost_values)
        std_dev = np.std(cost_values)
        
        # Range
        cost_range = max(cost_values) - min(cost_values)
        
        return {
            'Average Deviation from Mean': round(avg_deviation_from_mean, 2),
            'Max Deviation from Mean': round(max_deviation_from_mean, 2),
            'Average Deviation from Proportional': round(avg_deviation_from_prop, 2),
            'Standard Deviation': round(std_dev, 2),
            'Range (Max - Min)': round(cost_range, 2),
            'Variance': round(variance, 2)
        }
    
    def plot_comparison(self, methods_to_plot: List[str] = None):
        """Create a visual comparison of different allocation methods."""
        if methods_to_plot is None:
            methods_to_plot = ['Proportional', 'Quadratic (λ=1.0)', 'Progressive Cap']
        
        methods = {
            'Proportional': self.proportional_costs,
            'Equal Payment': self.equal_payment(),
            'Quadratic (λ=0.25)': self.quadratic_optimization(0.25),
            'Quadratic (λ=1.0)': self.quadratic_optimization(1.0),
            'Progressive Cap': self.progressive_cap()
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Costs by method
        room_ids = list(self.room_sizes.keys())
        x = np.arange(len(room_ids))
        width = 0.8 / len(methods_to_plot)
        
        for i, method_name in enumerate(methods_to_plot):
            costs = [methods[method_name][room_id] for room_id in room_ids]
            ax1.bar(x + i * width, costs, width, label=method_name)
        
        ax1.axhline(y=self.target_mean, color='red', linestyle='--', 
                   label=f'Target Mean (£{self.target_mean:.0f})')
        ax1.set_xlabel('Room')
        ax1.set_ylabel('Monthly Rent (£)')
        ax1.set_title('Rent Allocation by Method')
        ax1.set_xticks(x + width * (len(methods_to_plot) - 1) / 2)
        ax1.set_xticklabels(room_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviation from target mean
        for i, method_name in enumerate(methods_to_plot):
            deviations = [methods[method_name][room_id] - self.target_mean 
                         for room_id in room_ids]
            ax2.bar(x + i * width, deviations, width, label=method_name)
        
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Room')
        ax2.set_ylabel('Deviation from Target Mean (£)')
        ax2.set_title('Deviation from Target Mean by Method')
        ax2.set_xticks(x + width * (len(methods_to_plot) - 1) / 2)
        ax2.set_xticklabels(room_ids)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_method_summary(self, method_name: str, costs: Dict[str, float]):
        """Print a clean summary for a specific allocation method."""
        print(f"\n💰 {method_name.upper()} ALLOCATION")
        print("=" * 50)
        individual_costs = self.get_individual_room_costs(costs)
        
        for room_id in sorted(costs.keys()):
            total_cost = costs[room_id]
            individual_cost = individual_costs[room_id]
            room_size = self.room_sizes[room_id]
            deviation = total_cost - self.target_mean
            
            print(f"{room_id} ({room_size} sqm):")
            print(f"  Individual room: £{individual_cost:.2f}")
            print(f"  Shared Room 2:   £{self.shared_cost_per_person:.2f}")
            print(f"  TOTAL MONTHLY:   £{total_cost:.2f} ({deviation:+.0f} from £{self.target_mean} mean)")
            print()
        
        total_verification = sum(costs.values())
        print(f"✅ Total rent verification: £{total_verification:.2f} (target: £{self.total_rent})")
        
        # Fairness metrics
        metrics = self.calculate_fairness_metrics(costs)
        print(f"📊 Range: £{metrics['Range (Max - Min)']} | Std Dev: £{metrics['Standard Deviation']:.2f}")

# Example usage and demonstration
def main():
    """Demonstrate the rent optimizer with your house data."""
    
    # Your house data
    room_sizes = {
        'Room 1': 30.22,
        'Room 3': 19.16,
        'Room 4': 16.04,
        'Room 5': 15.66,
        'Room 6': 11.10
    }
    
    total_rent = 4110
    shared_room_size = 15.19  # Room 2
    target_mean = 822  # £4110 / 5 people
    
    # Initialize optimizer
    optimizer = RentOptimizer(room_sizes, total_rent, shared_room_size, target_mean)
    
    print("🏠 RENT OPTIMIZATION ANALYSIS")
    print("=" * 50)
    print(f"Total rent: £{total_rent}")
    print(f"Room 2 (shared): {shared_room_size} sqm → £{optimizer.shared_room_cost:.2f} total")
    print(f"Each person's share of Room 2: £{optimizer.shared_cost_per_person:.2f}")
    print(f"Remaining for individual rooms: £{optimizer.individual_rent:.2f}")
    print(f"Target mean per person: £{target_mean}")
    
    # Analyze desirability factors based on floor plan
    print("\n" + "="*70)
    print("🎯 DESIRABILITY ANALYSIS")
    print("="*70)
    
    # Define pub proximity based on floor plan analysis
    pub_proximity = {
        'Room 1': 'close',    # Basement - likely closest to pub noise
        'Room 2': 'medium',   # 1st floor - moderate distance
        'Room 3': 'medium',   # 2nd floor - moderate distance  
        'Room 4': 'far',      # 2nd floor with balcony - away from pub
        'Room 5': 'far',      # 3rd floor - highest, furthest
        'Room 6': 'far'       # 3rd floor - highest, furthest
    }
    
    factors = optimizer.print_desirability_analysis(pub_proximity)
    
    # Show desirability-adjusted allocations
    print("💡 DESIRABILITY-ADJUSTED ALLOCATIONS")
    print("=" * 70)
    print("These methods account for room quality, not just size:")
    print()
    
    # Size + Desirability weighted allocation
    size_desirability_30 = optimizer._calculate_size_desirability_weighted(0.3)
    optimizer.print_method_summary("Size + Desirability (30% weight)", size_desirability_30)
    
    size_desirability_50 = optimizer._calculate_size_desirability_weighted(0.5)
    optimizer.print_method_summary("Size + Desirability (50% weight)", size_desirability_50)
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDED METHODS WITH DETAILED BREAKDOWN")
    print("="*60)
    
    # Progressive Cap (most practical)
    progressive_costs = optimizer.progressive_cap()
    optimizer.print_method_summary("Progressive Cap (Recommended)", progressive_costs)
    
    # Quadratic balanced
    balanced_costs = optimizer.quadratic_optimization(1.0)
    optimizer.print_method_summary("Quadratic Balanced (λ=1.0)", balanced_costs)
    
    # Proportional for comparison
    optimizer.print_method_summary("Proportional (Baseline)", optimizer.proportional_costs)
    
    print("\n" + "="*60)
    print("📋 QUICK COMPARISON TABLE")
    print("="*60)
    
    # Quick comparison table including desirability
    methods_to_compare = {
        'Proportional (Size Only)': optimizer.proportional_costs,
        'Progressive Cap': progressive_costs,
        'Size + Desirability (30%)': size_desirability_30,
        'Size + Desirability (50%)': size_desirability_50,
        'Equal Payment': optimizer.equal_payment()
    }
    
    print(f"{'Method':<25} {'Room 1':<8} {'Room 3':<8} {'Room 4':<8} {'Room 5':<8} {'Room 6':<8}")
    print("-" * 73)
    
    for method_name, costs in methods_to_compare.items():
        print(f"{method_name:<25}", end="")
        for room_id in ['Room 1', 'Room 3', 'Room 4', 'Room 5', 'Room 6']:
            print(f"£{costs[room_id]:<7.0f}", end="")
        print()
    
    # Show desirability scores
    print(f"\n{'Desirability Scores':<25}", end="")
    for room_id in ['Room 1', 'Room 3', 'Room 4', 'Room 5', 'Room 6']:
        score = optimizer.desirability_scores.get(room_id, 0)
        print(f"{score:<8.1f}", end="")
    print()
    
    print("\n🎯 KEY INSIGHTS FROM FLOOR PLAN ANALYSIS")
    print("=" * 50)
    print("🏆 Most Desirable: Room 4 (closest to communal balcony, good light, low noise)")
    print("😐 Moderate: Rooms 3, 5 (decent floors, average features)")  
    print("😕 Less Desirable: Room 1 (basement, pub noise), Room 6 (small, many stairs)")
    print("🏠 Shared Features: Room 2 (→living room) + Balcony (communal)")
    print("🍺 Pub Impact: Room 1 most affected, upper floors protected")
    print()
    print("💰 RECOMMENDATIONS")
    print("=" * 50)
    print("• Size + Desirability (30%): Balances room size with quality factors")
    print("• Progressive Cap: Limits extreme costs, simpler to explain")
    print("• Size + Desirability (50%): Higher weight on room quality vs pure size")
    print("\n💡 All costs include your £116.29 share of Room 2 conversion!")
    
    # Create visualization (uncomment if you have matplotlib)
    # optimizer.plot_comparison(['Proportional', 'Progressive Cap', 'Quadratic (λ=1.0)'])
    
    return optimizer

if __name__ == "__main__":
    optimizer = main()
    
    # Quick demo of how to use specific methods
    print("\n" + "="*60)
    print("🚀 QUICK USAGE EXAMPLES")
    print("="*60)
    print("# Analyze room desirability based on floor plan")
    print("factors = optimizer.print_desirability_analysis()")
    print("# Desirability scores automatically set from floor plan analysis")
    print()
    print("# Get size + desirability allocation (recommended for your house)")
    print("desirability_costs = optimizer._calculate_size_desirability_weighted(0.3)")
    print("# Result:", {k: f"£{v:.0f}" for k, v in size_desirability_30.items()})
    print()
    print("# Compare pure size vs size+desirability")
    print("size_only = optimizer.proportional_costs")
    print("# Size only:", {k: f"£{v:.0f}" for k, v in optimizer.proportional_costs.items()})
    print("# With desirability:", {k: f"£{v:.0f}" for k, v in size_desirability_30.items()})
    print()
    print("# Key differences based on your floor plan:")
    print("# - Room 4 pays slightly more (closest to communal balcony + good light)")
    print("# - Room 1 pays less (basement + pub noise)")
    print("# - Room 6 pays less (small + many stairs)")
    print("# - Balcony is communal (like Room 2 → living room)")
    print()
    print("# Set custom desirability scores if you disagree:")
    print("custom_scores = {'Room 1': 4, 'Room 3': 7, 'Room 4': 9, 'Room 5': 6, 'Room 6': 5}")
    print("optimizer.set_desirability_scores(custom_scores)")
    print("custom_allocation = optimizer._calculate_size_desirability_weighted(0.4)")
    print()
    print("# Everyone's share of Room 2 (shared living room):")
    print(f"shared_cost_per_person = £{optimizer.shared_cost_per_person:.2f}")