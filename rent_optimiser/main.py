#!/usr/bin/env python3
"""
Streamlit app for rent optimization analysis.

A user-friendly interface for housemates to determine fair rent allocation
using multiple optimization methods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
import json

# Import your improved classes
from house import House, Room, DesirabilityFactors
from optimizers.proportional_optimizer import (
    ProportionalOptimizer, DesirabilityWeightedOptimizer, 
    EqualPaymentOptimizer, FloorAdjustedOptimizer,
    create_optimizer_from_streamlit_form, get_available_optimizers
)
from optimizers.base_optimizer import OptimizationResult, OptimizerManager

# Page configuration
st.set_page_config(
    page_title="Fair Rent Calculator",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'house' not in st.session_state:
        st.session_state.house = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = []
    if 'current_method' not in st.session_state:
        st.session_state.current_method = "Proportional by Size"


def create_sample_house() -> House:
    """Create a sample house with your data for demonstration."""
    room_data = [
        {
            'id': 'Room 1',
            'size': 30.22,
            'size_score': 5.0,
            'noise_level': 2.0,
            'floor_level': 0,
            'has_balcony': False,
            'has_ensuite': True,
            'distance_to_kitchen': 'close',
            'natural_light': 3.0,
            'overall_condition': 4.0
        },
        {
            'id': 'Room 3', 
            'size': 19.16,
            'size_score': 4.0,
            'noise_level': 4.0,
            'floor_level': 2,
            'has_balcony': False,
            'has_ensuite': False,
            'distance_to_kitchen': 'medium',
            'natural_light': 4.0,
            'overall_condition': 4.0
        },
        {
            'id': 'Room 4',
            'size': 16.04,
            'size_score': 3.0,
            'noise_level': 5.0,
            'floor_level': 2,
            'has_balcony': True,
            'has_ensuite': False,
            'distance_to_kitchen': 'medium',
            'natural_light': 5.0,
            'overall_condition': 4.0
        },
        {
            'id': 'Room 5',
            'size': 15.66,
            'size_score': 3.0,
            'noise_level': 5.0,
            'floor_level': 3,
            'has_balcony': False,
            'has_ensuite': False,
            'distance_to_kitchen': 'far',
            'natural_light': 4.0,
            'overall_condition': 3.0
        },
        {
            'id': 'Room 6',
            'size': 11.10,
            'size_score': 2.5,
            'noise_level': 5.0,
            'floor_level': 3,
            'has_balcony': False,
            'has_ensuite': False,
            'distance_to_kitchen': 'far',
            'natural_light': 3.0,
            'overall_condition': 3.0
        }
    ]
    
    shared_room_data = {'id': 'Room 2', 'size': 15.19}
    
    return House.create_from_streamlit_data(4110.0, room_data, shared_room_data)


def render_sidebar() -> Optional[House]:
    """Render the sidebar with house configuration options."""
    st.sidebar.header("🏠 Configure Your House")
    
    # Load sample data button
    if st.sidebar.button("📋 Load Sample House", type="secondary"):
        sample_house = create_sample_house()
        st.session_state.house = sample_house
        st.rerun()
    
    # Total rent input
    total_rent = st.sidebar.number_input(
        "💰 Total Monthly Rent (£)", 
        min_value=100, 
        max_value=20000, 
        value=4110 if st.session_state.house is None else st.session_state.house.total_rent,
        step=10
    )
    
    # Number of rooms
    st.sidebar.subheader("🚪 Individual Rooms")
    num_rooms = st.sidebar.number_input(
        "Number of individual rooms", 
        min_value=1, 
        max_value=10, 
        value=5 if st.session_state.house is None else len(st.session_state.house.individual_rooms)
    )
    
    # Room configuration
    rooms_data = []
    for i in range(num_rooms):
        with st.sidebar.expander(f"Room {i+1}", expanded=i < 3):
            # Get existing data if available
            existing_room = None
            if st.session_state.house and i < len(st.session_state.house.individual_rooms):
                existing_room = st.session_state.house.individual_rooms[i]
            
            # Room basic info
            room_id = st.text_input(
                "Room Name", 
                value=existing_room.room_id if existing_room else f"Room {i+1}",
                key=f"room_name_{i}"
            )
            
            room_size = st.number_input(
                "Size (sqm)", 
                min_value=1.0, 
                max_value=100.0,
                value=float(existing_room.size if existing_room else 15.0),
                step=0.1,
                key=f"room_size_{i}"
            )
            
            # Desirability factors
            st.write("**Desirability Factors**")
            
            size_score = st.select_slider(
                "Room Size Quality",
                options=[1, 2, 3, 4, 5],
                value=int(existing_room.desirability_factors.size_score if existing_room else 3),
                key=f"size_score_{i}",
                help="1=Very Small, 3=Average, 5=Very Large"
            )
            
            noise_level = st.select_slider(
                "Noise Level",
                options=[1, 2, 3, 4, 5],
                value=int(existing_room.desirability_factors.noise_level if existing_room else 3),
                key=f"noise_level_{i}",
                help="1=Very Noisy, 3=Moderate, 5=Very Quiet"
            )
            
            floor_level = st.selectbox(
                "Floor Level",
                options=[-1, 0, 1, 2, 3, 4],
                index=max(0, existing_room.desirability_factors.floor_level + 1 if existing_room else 2),
                key=f"floor_level_{i}",
                format_func=lambda x: {-1: "Basement", 0: "Ground", 1: "1st Floor", 2: "2nd Floor", 3: "3rd Floor", 4: "4th Floor"}[x]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                has_balcony = st.checkbox(
                    "Has Balcony",
                    value=existing_room.desirability_factors.has_balcony if existing_room else False,
                    key=f"balcony_{i}"
                )
                
                has_ensuite = st.checkbox(
                    "Has En-suite",
                    value=existing_room.desirability_factors.has_ensuite if existing_room else False,
                    key=f"ensuite_{i}"
                )
            
            with col2:
                distance_to_kitchen = st.selectbox(
                    "Kitchen Distance",
                    options=["close", "medium", "far"],
                    index=["close", "medium", "far"].index(existing_room.desirability_factors.distance_to_kitchen if existing_room else "medium"),
                    key=f"kitchen_dist_{i}"
                )
                
                natural_light = st.select_slider(
                    "Natural Light",
                    options=[1, 2, 3, 4, 5],
                    value=int(existing_room.desirability_factors.natural_light if existing_room else 3),
                    key=f"light_{i}"
                )
            
            rooms_data.append({
                'id': room_id,
                'size': room_size,
                'size_score': float(size_score),
                'noise_level': float(noise_level),
                'floor_level': floor_level,
                'has_balcony': has_balcony,
                'has_ensuite': has_ensuite,
                'distance_to_kitchen': distance_to_kitchen,
                'natural_light': float(natural_light),
                'overall_condition': 3.0  # Default
            })
    
    # Shared room configuration
    st.sidebar.subheader("🛋️ Shared Spaces")
    include_shared = st.sidebar.checkbox(
        "Include shared room in calculation",
        value=st.session_state.house.shared_room is not None if st.session_state.house else True
    )
    
    shared_room_data = None
    if include_shared:
        shared_id = st.sidebar.text_input(
            "Shared Room Name",
            value=st.session_state.house.shared_room.room_id if st.session_state.house and st.session_state.house.shared_room else "Living Room"
        )
        shared_size = st.sidebar.number_input(
            "Shared Room Size (sqm)",
            min_value=1.0,
            max_value=100.0,
            value=float(st.session_state.house.shared_room.size if st.session_state.house and st.session_state.house.shared_room else 20.0),
            step=0.1
        )
        shared_room_data = {'id': shared_id, 'size': shared_size}
    
    # Create house button
    if st.sidebar.button("🔄 Update House Configuration", type="primary"):
        try:
            # Validate inputs
            errors = []
            
            # Check for duplicate room names
            room_names = [room['id'].strip() for room in rooms_data]
            if len(room_names) != len(set(room_names)):
                errors.append("Room names must be unique")
            
            # Check for empty room names
            if any(not name.strip() for name in room_names):
                errors.append("Room names cannot be empty")
            
            if errors:
                for error in errors:
                    st.sidebar.error(error)
            else:
                # Create house
                house = House.create_from_streamlit_data(
                    total_rent, rooms_data, shared_room_data
                )
                st.session_state.house = house
                st.session_state.optimization_results = []  # Clear previous results
                st.sidebar.success("✅ House configuration updated!")
                st.rerun()
                
        except Exception as e:
            st.sidebar.error(f"Error creating house: {str(e)}")
    
    return st.session_state.house


def render_method_selection() -> Dict:
    """Render method selection and parameter configuration."""
    st.subheader("⚙️ Choose Optimization Method")
    
    available_methods = list(get_available_optimizers().keys())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        method_name = st.selectbox(
            "Select Method",
            available_methods,
            index=available_methods.index(st.session_state.current_method) if st.session_state.current_method in available_methods else 0,
            help="Choose how to allocate rent among rooms"
        )
    
    with col2:
        auto_calculate = st.checkbox(
            "Auto-calculate",
            value=True,
            help="Automatically recalculate when settings change"
        )
    
    # Method-specific parameters
    parameters = {}
    
    if method_name == "Size + Desirability":
        parameters['desirability_weight'] = st.slider(
            "Desirability Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="0 = Pure size-based, 1 = Maximum desirability adjustment"
        )
        
        st.info("💡 **How it works:** Better rooms (higher desirability) pay slightly less, worse rooms pay slightly more, while still considering room size.")
    
    elif method_name == "Floor Level Adjusted":
        parameters['floor_adjustment_rate'] = st.slider(
            "Floor Adjustment Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="How much to adjust rent based on floor level"
        )
        
        st.info("💡 **How it works:** Higher floors typically pay less (quieter, better views), lower floors pay more.")
    
    elif method_name == "Proportional by Size":
        st.info("💡 **How it works:** Larger rooms pay proportionally more. The simplest and most straightforward method.")
    
    elif method_name == "Equal Payment":
        st.info("💡 **How it works:** Everyone pays exactly the same amount regardless of room size or quality.")
    
    # Store current method
    st.session_state.current_method = method_name
    
    return {
        'method_name': method_name,
        'parameters': parameters,
        'auto_calculate': auto_calculate
    }


def run_optimization(house: House, method_config: Dict) -> OptimizationResult:
    """Run optimization and return results."""
    try:
        optimizer = create_optimizer_from_streamlit_form(
            method_config['method_name'],
            method_config['parameters']
        )
        
        result = optimizer.get_optimization_result(house)
        return result
        
    except Exception as e:
        st.error(f"Error running optimization: {str(e)}")
        return None


def render_results(house: House, result: OptimizationResult):
    """Render the optimization results."""
    if not result:
        st.warning("⚠️ Configure your house and select a method to see results.")
        return
    
    # Key metrics
    st.subheader("💰 Rent Allocation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Rent",
            f"£{house.total_rent:,.0f}",
            help="Total monthly rent for the house"
        )
    
    with col2:
        st.metric(
            "Average per Person",
            f"£{house.target_mean:.0f}",
            help="Equal split amount for comparison"
        )
    
    with col3:
        fairness_score = result.fairness_metrics.get('Fairness_Score', 0)
        st.metric(
            "Fairness Score",
            f"{fairness_score:.0f}/100",
            help="Higher score = more fair allocation"
        )
    
    with col4:
        std_dev = result.fairness_metrics.get('Standard_Deviation', 0)
        st.metric(
            "Variation",
            f"±£{std_dev:.0f}",
            help="How much costs vary between rooms"
        )
    
    # Main results table
    allocation_data = []
    for room in house.individual_rooms:
        total_cost = result.allocation[room.room_id]
        individual_cost = total_cost - house.shared_cost_per_person
        deviation = total_cost - house.target_mean
        
        allocation_data.append({
            'Room': room.room_id,
            'Size (sqm)': f"{room.size:.1f}",
            'Desirability': f"{room.desirability_score:.1f}/5",
            'Individual Cost': f"£{individual_cost:.0f}",
            'Shared Portion': f"£{house.shared_cost_per_person:.0f}",
            'Total Monthly': f"£{total_cost:.0f}",
            'vs. Average': f"{deviation:+.0f}",
            'Per sqm': f"£{total_cost/room.size:.1f}"
        })
    
    df = pd.DataFrame(allocation_data)
    
    # Style the dataframe
    def highlight_deviation(val):
        if 'vs. Average' in val.name:
            try:
                num_val = float(val.str.replace('£', '').str.replace('+', ''))
                if num_val > 50:
                    return ['background-color: #ffebee'] * len(val)  # Light red
                elif num_val < -50:
                    return ['background-color: #e8f5e8'] * len(val)  # Light green
            except:
                pass
        return [''] * len(val)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization
    render_visualization(house, result)


def render_visualization(house: House, result: OptimizationResult):
    """Create interactive visualizations of the results."""
    st.subheader("📊 Visual Analysis")
    
    # Prepare data for plotting
    plot_data = []
    for room in house.individual_rooms:
        total_cost = result.allocation[room.room_id]
        deviation = total_cost - house.target_mean
        
        plot_data.append({
            'Room': room.room_id,
            'Total_Cost': total_cost,
            'Size': room.size,
            'Desirability': room.desirability_score,
            'Deviation': deviation,
            'Cost_per_sqm': total_cost / room.size
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of total costs
        fig_bar = px.bar(
            df_plot,
            x='Room',
            y='Total_Cost',
            title='Monthly Rent by Room',
            color='Deviation',
            color_continuous_scale='RdYlGn_r',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Deviation': 'vs Average (£)'}
        )
        fig_bar.add_hline(
            y=house.target_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text="Average (Equal Split)"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Scatter plot: Size vs Cost
        fig_scatter = px.scatter(
            df_plot,
            x='Size',
            y='Total_Cost',
            size='Desirability',
            color='Room',
            title='Room Size vs Monthly Rent',
            labels={'Size': 'Room Size (sqm)', 'Total_Cost': 'Monthly Rent (£)'},
            hover_data=['Cost_per_sqm']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_comparison_mode(house: House):
    """Render comparison between multiple methods."""
    st.subheader("🔍 Compare Multiple Methods")
    
    if st.button("🚀 Run All Methods"):
        with st.spinner("Running all optimization methods..."):
            methods_to_run = [
                ("Proportional by Size", {}),
                ("Size + Desirability", {"desirability_weight": 0.3}),
                ("Size + Desirability", {"desirability_weight": 0.5}),
                ("Equal Payment", {}),
                ("Floor Level Adjusted", {"floor_adjustment_rate": 0.15}),
            ]
            
            comparison_results = []
            for method_name, params in methods_to_run:
                try:
                    optimizer = create_optimizer_from_streamlit_form(method_name, params)
                    result = optimizer.get_optimization_result(house)
                    comparison_results.append(result)
                except Exception as e:
                    st.error(f"Error with {method_name}: {str(e)}")
            
            if comparison_results:
                # Create comparison dataframe
                comparison_data = []
                for result in comparison_results:
                    comparison_data.append({
                        'Method': result.method_name,
                        'Fairness Score': result.fairness_metrics['Fairness_Score'],
                        'Std Deviation': result.fairness_metrics['Standard_Deviation'],
                        'Range': result.fairness_metrics['Range_Max_Min'],
                        'Gini Coefficient': result.fairness_metrics['Gini_Coefficient']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                st.write("**Method Comparison Summary**")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Recommend best method
                best_method = comparison_df.loc[comparison_df['Fairness Score'].idxmax()]
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>🎯 Recommended Method</h4>
                    <p><strong>{best_method['Method']}</strong> has the highest fairness score ({best_method['Fairness Score']:.0f}/100).</p>
                    <p>This method provides the best balance of fairness while considering your room characteristics.</p>
                </div>
                """, unsafe_allow_html=True)


def render_export_options(house: House, result: OptimizationResult):
    """Render options to export/save results."""
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Summary"):
            summary_text = f"""
# Rent Allocation Summary

**House Details:**
- Total Rent: £{house.total_rent:,.0f}
- Number of People: {house.num_people}
- Average per Person: £{house.target_mean:.0f}

**Method Used:** {result.method_name}
**Fairness Score:** {result.fairness_metrics['Fairness_Score']:.0f}/100

**Allocation:**
"""
            for room in house.individual_rooms:
                total_cost = result.allocation[room.room_id]
                summary_text += f"- {room.room_id}: £{total_cost:.0f}\n"
            
            st.text_area("Summary", summary_text, height=300)
    
    with col2:
        if st.button("📊 Download CSV"):
            # Create detailed CSV data
            csv_data = []
            for room in house.individual_rooms:
                total_cost = result.allocation[room.room_id]
                individual_cost = total_cost - house.shared_cost_per_person
                
                csv_data.append({
                    'Room_ID': room.room_id,
                    'Size_sqm': room.size,
                    'Desirability_Score': room.desirability_score,
                    'Individual_Cost': individual_cost,
                    'Shared_Cost': house.shared_cost_per_person,
                    'Total_Monthly_Cost': total_cost,
                    'Method': result.method_name
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"rent_allocation_{result.method_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔗 Share Configuration"):
            # Create shareable configuration
            config = {
                'house': house.to_dict(),
                'method': result.method_name,
                'parameters': result.parameters
            }
            
            config_json = json.dumps(config, indent=2)
            st.text_area("Configuration (shareable)", config_json, height=200)


def main():
    """Main Streamlit app function."""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<p class="main-header">🏠 Fair Rent Calculator</p>', unsafe_allow_html=True)
    st.markdown("**Determine fair rent allocation for your shared house using multiple optimization methods.**")
    
    # Sidebar configuration
    house = render_sidebar()
    
    if house is None:
        st.info("👈 Configure your house details in the sidebar to get started, or load the sample data.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Method selection and results
        method_config = render_method_selection()
        
        # Run optimization
        result = None
        if method_config['auto_calculate']:
            result = run_optimization(house, method_config)
        else:
            if st.button("🔥 Calculate Rent Allocation", type="primary"):
                result = run_optimization(house, method_config)
        
        if result:
            render_results(house, result)
    
    with col2:
        # House summary
        st.subheader("🏠 House Summary")
        
        st.metric("Total Rent", f"£{house.total_rent:,.0f}")
        st.metric("Number of People", house.num_people)
        st.metric("Total Area", f"{house.total_area:.1f} sqm")
        
        if house.shared_room:
            st.metric("Shared Room", f"{house.shared_room.size:.1f} sqm")
            st.metric("Shared Cost/Person", f"£{house.shared_cost_per_person:.0f}")
        
        # Room summary
        st.write("**Rooms:**")
        for room in house.individual_rooms:
            st.write(f"• {room.room_id}: {room.size:.1f} sqm (Desirability: {room.desirability_score:.1f}/5)")
    
    # Additional features
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Compare Methods", "💾 Export", "ℹ️ About"])
    
    with tab1:
        render_comparison_mode(house)
    
    with tab2:
        if result:
            render_export_options(house, result)
        else:
            st.info("Calculate rent allocation first to see export options.")
    
    with tab3:
        st.markdown("""
        ### About This Tool
        
        This rent calculator helps housemates determine fair rent allocation using multiple optimization methods:
        
        **🏠 Methods Available:**
        - **Proportional by Size**: Larger rooms pay more (baseline)
        - **Size + Desirability**: Adjusts for room quality factors
        - **Equal Payment**: Everyone pays the same amount
        - **Floor Level Adjusted**: Higher floors typically pay less
        
        **📊 Fairness Metrics:**
        - **Fairness Score**: Overall fairness rating (0-100)
        - **Standard Deviation**: How much costs vary
        - **Gini Coefficient**: Measure of inequality
        
        **💡 Tips:**
        - Start with "Proportional by Size" as a baseline
        - Use "Size + Desirability" if rooms have different qualities
        - Compare multiple methods to find what works best
        - The fairness score helps identify the most balanced approach
        
        **🔧 Built with:**
        - Streamlit for the interface
        - Plotly for interactive charts
        - Custom optimization algorithms
        """)


if __name__ == "__main__":
    main()