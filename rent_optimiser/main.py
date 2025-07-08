#!/usr/bin/env python3
"""
Main script for running rent optimization analysis.

This script demonstrates the complete workflow using the new OOP structure.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from house import House
from optimizers.proportional_optimizer import (
    ProportionalOptimizer, 
    DesirabilityWeightedOptimizer, 
    EqualPaymentOptimizer
)
from optimizers.quadratic_optimizer import (
    QuadraticOptimizer,
    ProgressiveCapOptimizer,
    MinMaxDeviationOptimizer,
    CustomWeightedOptimizer
)
from utils.comparison import OptimizationComparison


def create_house_from_your_data() -> House:
    """Create house object with your specific data."""
    # Your house data
    room_data = {
        'Room 1': 30.22,  # a little bigger bc of oriel
        'Room 3': 19.16,  # a little bigger bc of oriel  
        'Room 4': 16.04,  # complex shape calculation from your data
        'Room 5': 15.66,  # 5,22m x 3m
        'Room 6': 11.10   # smallest room
    }
    
    # Create house object
    house = House.create_from_data(
        room_data=room_data,
        total_rent=4110,
        shared_room_id="Room 2",
        shared_room_size=15.19  # a little bigger bc of oriel
    )
    
    # Calculate desirability factors for all rooms
    pub_proximity = {
        'Room 1': 'close',    # Basement - likely closest to pub noise
        'Room 2': 'medium',   # 1st floor - moderate distance
        'Room 3': 'medium',   # 2nd floor - moderate distance  
        'Room 4': 'far',      # 2nd floor with balcony - away from pub
        'Room 5': 'far',      # 3rd floor - highest, furthest
        'Room 6': 'far'       # 3rd floor - highest, furthest
    }
    
    house.calculate_desirability_for_all_rooms()
    
    return house


def run_all_optimizations(house: House) -> OptimizationComparison:
    """Run all optimization methods and return comparison object."""
    
    # Create comparison object
    comparison = OptimizationComparison(house, output_dir="results")
    
    # Add all optimizers
    optimizers = [
        # Basic methods
        ProportionalOptimizer(house),
        EqualPaymentOptimizer(house),
        
        # Desirability-weighted methods
        DesirabilityWeightedOptimizer(house, desirability_weight=0.3),
        DesirabilityWeightedOptimizer(house, desirability_weight=0.5),
        
        # Quadratic optimization (different lambda values)
        QuadraticOptimizer(house, lambda_param=0.25),
        QuadraticOptimizer(house, lambda_param=1.0),
        QuadraticOptimizer(house, lambda_param=2.0),
        
        # Progressive cap methods
        ProgressiveCapOptimizer(house, max_above_mean=100, max_below_mean=150),
        ProgressiveCapOptimizer(house, max_above_mean=75, max_below_mean=100),
        
        # Advanced methods
        MinMaxDeviationOptimizer(house),
        
        # Custom weighted (example with manual adjustments)
        CustomWeightedOptimizer(
            house, 
            size_weight=0.6, 
            desirability_weight=0.3,
            custom_adjustments={'Room 1': 0.2, 'Room 4': 0.8}  # Room 1 less desirable, Room 4 more
        )
    ]
    
    comparison.add_optimizers(optimizers)
    
    return comparison


def main():
    """Main function that runs the complete analysis."""
    
    print("🏠 RENT OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print("Running comprehensive analysis with your house data...")
    
    # Create house object
    house = create_house_from_your_data()
    
    # Print house summary
    house.print_summary()
    house.print_desirability_analysis()
    
    # Run all optimizations
    print("\n⚙️ RUNNING ALL OPTIMIZATION METHODS...")
    print("=" * 50)
    
    comparison = run_all_optimizations(house)
    results = comparison.run_comparison()
    
    # Print comparison summary
    comparison.print_comparison_summary()
    
    # Export results to CSV
    print(f"\n💾 EXPORTING RESULTS...")
    print("=" * 30)
    exported_files = comparison.export_to_csv("rent_price_methods")
    
    # Get recommendations
    print(f"\n🎯 METHOD RECOMMENDATIONS")
    print("=" * 40)
    recommendations = comparison.get_recommendations()
    for criteria, method in recommendations.items():
        print(f"{criteria}: {method}")
    
    # Create visualization (optional - requires matplotlib)
    try:
        print(f"\n📊 CREATING VISUALIZATIONS...")
        print("=" * 35)
        comparison.create_visualization(save_plots=True)
    except ImportError:
        print("⚠️ Matplotlib not available - skipping visualization")
    
    # Detailed analysis of top methods
    print(f"\n🔍 DETAILED ANALYSIS OF RECOMMENDED METHODS")
    print("=" * 60)
    
    # Get the most balanced method (Progressive Cap) and print details
    progressive_optimizer = ProgressiveCapOptimizer(house)
    progressive_optimizer.print_summary()
    
    # Get the desirability-weighted method and print details  
    desirability_optimizer = DesirabilityWeightedOptimizer(house, 0.3)
    desirability_optimizer.print_summary()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print("=" * 25)
    print("Check the 'results' folder for detailed CSV outputs.")
    print("\nKey files generated:")
    for file_type, file_path in exported_files.items():
        print(f"  • {file_type}: {file_path.name}")


def run_single_method_example():
    """Example of running just one optimization method (base case)."""
    
    print("\n📝 SINGLE METHOD EXAMPLE (BASE CASE)")
    print("=" * 50)
    
    # Create house
    house = create_house_from_your_data()
    
    # Run single optimizer
    optimizer = ProgressiveCapOptimizer(house)
    
    # Use comparison utility for single optimizer (base case)
    comparison = OptimizationComparison(house)
    result_df = comparison.handle_single_optimizer(optimizer)
    
    print("Single method results:")
    print(result_df[['Room_ID', 'Size_sqm', 'Total_Monthly_Cost', 'Deviation_from_Mean']])


if __name__ == "__main__":
    # Run complete analysis
    main()
    
    # Uncomment to see single method example
    # run_single_method_example()