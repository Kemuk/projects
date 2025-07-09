#!/usr/bin/env python3
"""
Streamlined Streamlit app for rent optimization using Quadratic Optimization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional
import json
import zipfile
import io

# Import your classes
from house import House, Room, DesirabilityFactors
from optimizers.quadratic_optimizer import QuadraticOptimizer
from optimizers.base_optimizer import OptimizationResult
from utils.comparison import OptimizationComparison

# Page configuration
st.set_page_config(
    page_title="Fair Rent Calculator",
    page_icon="🏠",
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
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .lambda-description {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
        font-style: italic;
    }
    .export-button {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'house' not in st.session_state:
        try:
            st.session_state.house = House.create_your_house()
        except Exception as e:
            st.error(f"Error creating house: {e}")
            # Fallback: create a simple house manually
            st.session_state.house = create_fallback_house()
    if 'lambda_param' not in st.session_state:
        st.session_state.lambda_param = 5.0
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = []


def create_fallback_house() -> House:
    """Create a fallback house if the main creation fails."""
    try:
        # Create rooms manually as a fallback
        rooms_data = [
            ('Room 1', 30.22, 5.0, 2.0, 4.0),
            ('Room 3', 19.16, 4.0, 4.0, 3.0),
            ('Room 4', 16.04, 3.0, 5.0, 3.0),
            ('Room 5', 15.66, 3.0, 5.0, 2.0),
            ('Room 6', 11.10, 2.0, 5.0, 1.0),
        ]
        
        individual_rooms = []
        for room_id, size, size_score, noise_level, accessibility in rooms_data:
            factors = DesirabilityFactors(
                size_score=size_score,
                noise_level=noise_level,
                accessibility=accessibility
            )
            room = Room(room_id=room_id, size=size, desirability_factors=factors)
            individual_rooms.append(room)
        
        shared_room = Room('Room 2', 15.19, is_shared=True)
        return House(individual_rooms, shared_room, 4110.0)
        
    except Exception as e:
        st.error(f"Critical error creating house: {e}")
        # Return None if everything fails
        return None


def render_sidebar() -> House:
    """Render the sidebar with optimization settings and room configuration."""
    
    # Validate house object first
    if not st.session_state.house or not hasattr(st.session_state.house, 'total_rent'):
        st.sidebar.error("⚠️ House data not loaded properly")
        if st.sidebar.button("🔄 Reload House Data"):
            st.session_state.house = create_fallback_house()
            st.rerun()
        return None
    
    st.sidebar.header("🎛️ Optimization Settings")
    
    # Lambda parameter slider with conceptual endpoints
    st.sidebar.markdown("**Allocation Method:**")
    lambda_param = st.sidebar.slider(
        "Equal Cost ← → Cost per sqm",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.lambda_param,
        step=0.5,
        help="Left: Everyone pays the same amount | Right: Rent proportional to room value"
    )
    
    # Update session state
    st.session_state.lambda_param = lambda_param
    
    # Show conceptual description instead of just lambda
    if lambda_param == 0:
        description = "🟰 Equal Cost (everyone pays same)"
    elif lambda_param >= 9.5:
        description = "📏 Cost per sqm (proportional)"
    else:
        # Calculate percentage towards proportional
        percentage = (lambda_param / 10) * 100
        description = f"⚖️ {percentage:.0f}% towards proportional"
    
    st.sidebar.markdown(f'<div class="lambda-description">{description}</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # House configuration
    st.sidebar.header("🏠 House Configuration")
    
    # Quick load defaults
    if st.sidebar.button("📋 Load Your House Data", type="secondary", help="Load your house with default settings"):
        try:
            st.session_state.house = House.create_your_house()
        except Exception as e:
            st.sidebar.error(f"Error loading house: {e}")
            st.session_state.house = create_fallback_house()
        st.rerun()
    
    # Total rent
    total_rent = st.sidebar.number_input(
        "💰 Total Monthly Rent (£)",
        min_value=100,
        max_value=20000,
        value=int(st.session_state.house.total_rent),
        step=10
    )
    
    # Shared room configuration
    st.sidebar.subheader("🛋️ Shared Living Space")
    shared_room_size = st.sidebar.number_input(
        "Shared Room Size (sqm)",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.house.shared_room.size if st.session_state.house.shared_room else 15.19),
        step=0.1,
        help="Size of the shared living space (Room 2)"
    )
    
    st.sidebar.markdown("---")
    
    # Room configuration - more prominent
    st.sidebar.subheader("🚪 Configure Individual Rooms")
    st.sidebar.markdown("*Adjust room characteristics:*")
    
    rooms_data = []
    for i, room in enumerate(st.session_state.house.individual_rooms):
        st.sidebar.markdown(f"**{room.room_id}**")
        
        # Room size (editable)
        room_size = st.sidebar.number_input(
            f"Size (sqm)",
            min_value=1.0,
            max_value=100.0,
            value=float(room.size),
            step=0.1,
            key=f"size_{i}",
            help=f"Physical size of {room.room_id}"
        )
        
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            size_score = st.select_slider(
                "Feel",
                options=[1, 2, 3, 4, 5],
                value=int(room.desirability_factors.size_score),
                key=f"feel_{i}",
                help="How spacious does this room feel? (50% weight)"
            )
        
        with col2:
            noise_level = st.select_slider(
                "Quiet",
                options=[1, 2, 3, 4, 5],
                value=int(room.desirability_factors.noise_level),
                key=f"quiet_{i}",
                help="How quiet is this room? (40% weight)"
            )
        
        with col3:
            accessibility = st.select_slider(
                "Access",
                options=[1, 2, 3, 4, 5],
                value=int(room.desirability_factors.accessibility),
                key=f"access_{i}",
                help="How convenient to kitchen/bathroom? (10% weight)"
            )
        
        # Show current desirability score
        current_factors = DesirabilityFactors(size_score, noise_level, accessibility)
        current_score = current_factors.calculate_overall_score()
        st.sidebar.caption(f"Score: {current_score:.1f}/5 | £{(total_rent/6)*(room_size/15):.0f}/month est.")
        st.sidebar.markdown("---")
        
        rooms_data.append({
            'room_id': room.room_id,
            'size': room_size,
            'size_score': float(size_score),
            'noise_level': float(noise_level),
            'accessibility': float(accessibility)
        })
    
    # Auto-update or manual update
    auto_update = st.sidebar.checkbox("🔄 Auto-update results", value=True, help="Update results as you change settings")
    
    # Update house when settings change (auto or manual)
    should_update = auto_update
    if not auto_update:
        should_update = st.sidebar.button("📊 Calculate with New Settings", type="primary")
    
    if should_update:
        # Check if any settings actually changed
        settings_changed = False
        
        # Check room changes
        for i, room_data in enumerate(rooms_data):
            room = st.session_state.house.individual_rooms[i]
            if (room_data['size'] != room.size or
                room_data['size_score'] != room.desirability_factors.size_score or
                room_data['noise_level'] != room.desirability_factors.noise_level or
                room_data['accessibility'] != room.desirability_factors.accessibility):
                settings_changed = True
                break
        
        # Check other changes
        if (total_rent != st.session_state.house.total_rent or
            shared_room_size != st.session_state.house.shared_room.size):
            settings_changed = True
        
        if settings_changed:
            # Create updated rooms
            individual_rooms = []
            for room_data in rooms_data:
                factors = DesirabilityFactors(
                    size_score=room_data['size_score'],
                    noise_level=room_data['noise_level'],
                    accessibility=room_data['accessibility']
                )
                room = Room(
                    room_id=room_data['room_id'],
                    size=room_data['size'],
                    desirability_factors=factors
                )
                individual_rooms.append(room)
            
            # Create updated shared room
            shared_room = Room('Room 2', shared_room_size, is_shared=True)
            
            # Update house
            st.session_state.house = House(individual_rooms, shared_room, float(total_rent))
            
            if not auto_update:
                st.sidebar.success("✅ Settings updated!")
                st.rerun()
    
    return st.session_state.house


def run_optimization(house: House, lambda_param: float) -> OptimizationResult:
    """Run quadratic optimization."""
    try:
        optimizer = QuadraticOptimizer(lambda_param)
        result = optimizer.get_optimization_result(house)
        return result
    except Exception as e:
        st.error(f"Error running optimization: {str(e)}")
        return None


def render_results(house: House, result: OptimizationResult):
    """Render the optimization results."""
    if not result:
        st.warning("⚠️ Unable to calculate results. Please check your settings.")
        return
    
    # Key metrics row
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
        range_val = result.fairness_metrics.get('Range_Max_Min', 0)
        st.metric(
            "Range",
            f"£{range_val:.0f}",
            help="Difference between highest and lowest rent"
        )
    
    with col4:
        std_dev = result.fairness_metrics.get('Standard_Deviation', 0)
        gini = result.fairness_metrics.get('Gini_Coefficient', 0)
        st.metric(
            "Spread",
            f"±£{std_dev:.0f}",
            help=f"Standard deviation of rents. Gini coefficient: {gini:.3f} (0=perfect equality, 1=maximum inequality)"
        )
    
    # Results table
    st.subheader("📊 Individual Rent Breakdown")
    
    # Prepare data for table
    table_data = []
    for room in house.individual_rooms:
        total_cost = result.allocation[room.room_id]
        individual_cost = total_cost - house.shared_cost_per_person
        deviation = total_cost - house.target_mean
        
        table_data.append({
            'Room': room.room_id,
            'Size (sqm)': f"{room.size:.1f}",
            'Desirability': f"{room.desirability_score:.1f}/5",
            'Monthly Rent': f"£{total_cost:.0f}",
            'vs. Average': f"{deviation:+.0f}",
            'Per sqm': f"£{total_cost/room.size:.1f}"
        })
    
    df = pd.DataFrame(table_data)
    
    # Display the table with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    render_visualizations(house, result)


def render_visualizations(house: House, result: OptimizationResult):
    """Create dynamic visualizations."""
    st.subheader("📈 Visual Analysis")
    
    # Prepare data
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
            'Cost_per_sqm': total_cost / room.size,
            'Size_Score': room.desirability_factors.size_score,
            'Noise_Level': room.desirability_factors.noise_level,
            'Accessibility': room.desirability_factors.accessibility
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of total costs with dynamic coloring
        fig_bar = px.bar(
            df_plot,
            x='Room',
            y='Total_Cost',
            title='Monthly Rent by Room',
            color='Deviation',
            color_continuous_scale='RdYlGn_r',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Deviation': 'vs Average (£)'},
            hover_data={
                'Size': ':.1f',
                'Desirability': ':.1f',
                'Cost_per_sqm': ':.1f'
            }
        )
        
        # Add average line
        fig_bar.add_hline(
            y=house.target_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text="Average (Equal Split)"
        )
        
        fig_bar.update_layout(
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Scatter plot: Desirability vs Cost
        fig_scatter = px.scatter(
            df_plot,
            x='Desirability',
            y='Total_Cost',
            size='Size',
            color='Room',
            title='Room Desirability vs Monthly Rent',
            labels={
                'Desirability': 'Desirability Score (1-5)', 
                'Total_Cost': 'Monthly Rent (£)',
                'Size': 'Room Size (sqm)'
            },
            hover_data={
                'Size_Score': True,
                'Noise_Level': True,
                'Accessibility': True,
                'Cost_per_sqm': ':.1f'
            }
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_visualizations(house: House, result: OptimizationResult):
    """Create dynamic visualizations."""
    st.subheader("📈 Visual Analysis")
    
    # Prepare data
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
            'Cost_per_sqm': total_cost / room.size,
            'Size_Score': room.desirability_factors.size_score,
            'Noise_Level': room.desirability_factors.noise_level,
            'Accessibility': room.desirability_factors.accessibility
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of total costs with dynamic coloring
        fig_bar = px.bar(
            df_plot,
            x='Room',
            y='Total_Cost',
            title='Monthly Rent by Room',
            color='Deviation',
            color_continuous_scale='RdYlGn_r',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Deviation': 'vs Average (£)'},
            hover_data={
                'Size': ':.1f',
                'Desirability': ':.1f',
                'Cost_per_sqm': ':.1f'
            }
        )
        
        # Add average line
        fig_bar.add_hline(
            y=house.target_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text="Average (Equal Split)"
        )
        
        fig_bar.update_layout(
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Scatter plot: Desirability vs Cost
        fig_scatter = px.scatter(
            df_plot,
            x='Desirability',
            y='Total_Cost',
            size='Size',
            color='Room',
            title='Room Desirability vs Monthly Rent',
            labels={
                'Desirability': 'Desirability Score (1-5)', 
                'Total_Cost': 'Monthly Rent (£)',
                'Size': 'Room Size (sqm)'
            },
            hover_data={
                'Size_Score': True,
                'Noise_Level': True,
                'Accessibility': True,
                'Cost_per_sqm': ':.1f'
            }
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)


def main():
    """Main Streamlit app function."""
    # Initialize session state
    initialize_session_state()
    
    # Header with export in top right
    col_title, col_export = st.columns([4, 1])
    
    with col_title:
        st.markdown('<p class="main-header">🏠 Fair Rent Calculator</p>', unsafe_allow_html=True)
        st.markdown("**Optimize rent allocation using quadratic optimization - balance equality with proportionality**")
    
    with col_export:
        st.markdown("### ")  # Space for alignment
        if st.button("💾 Export Results", type="secondary", help="Export current results"):
            render_export_modal()
    
    # Sidebar configuration
    house = render_sidebar()
    
    # Check if house is valid
    if not house:
        st.error("🏠 Unable to load house configuration. Please check the sidebar to reload.")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Run current optimization
        result = run_optimization(house, st.session_state.lambda_param)
        
        if result:
            # Auto-run comparison analysis first to get data
            comparison = OptimizationComparison(house)
            st.session_state.comparison_results = comparison
            
            with st.spinner("Running analysis..."):
                lambda_values = list(range(11))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                comparison_data = comparison.run_lambda_comparison(lambda_values)
            
            # Main content tabs
            tab1, tab2, tab3, tab4 = st.tabs(["💰 Current Results", "📊 Fairness Analysis", "📋 Cost Summary", "ℹ️ Methodology"])
            
            with tab1:
                render_results(house, result)
                
                # Quick comparison summary
                st.markdown("---")
                st.subheader("📈 Quick Comparison Summary")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    current_costs = comparison_data['detailed_results']
                    min_cost = current_costs['Total_Cost'].min()
                    max_cost = current_costs['Total_Cost'].max()
                    st.metric(
                        "Range Across Methods",
                        f"£{min_cost:.0f} - £{max_cost:.0f}",
                        help="Lowest to highest room cost across all settings"
                    )
                
                with col_b:
                    fairness_df = comparison_data['fairness_metrics']
                    best_gini_lambda = int(fairness_df.loc[fairness_df['Gini_Coefficient'].idxmin(), 'Lambda'])
                    st.metric(
                        "Most Equal Setting",
                        f"Balance = {best_gini_lambda}",
                        help="Setting that produces the most equal distribution"
                    )
                
                with col_c:
                    current_lambda = st.session_state.lambda_param
                    # Round to nearest integer since we have exact data for all integers
                    rounded_lambda = int(round(current_lambda))
                    current_gini = fairness_df[fairness_df['Lambda'] == rounded_lambda]['Gini_Coefficient'].iloc[0]
                    
                    st.metric(
                        "Current Equality",
                        f"Gini: {current_gini:.3f}",
                        help="Lower = more equal (0=perfect equality, 1=max inequality)"
                    )
            
            with tab2:
                comparison.render_fairness_tab()
            
            with tab3:
                comparison.render_summary_tab()
            
            with tab4:
                render_methodology()
    
    with col2:
        # House summary
        st.subheader("🏠 Current Setup")
        
        st.metric("Total Rent", f"£{house.total_rent:,.0f}")
        st.metric("People", house.num_people)
        
        # Lambda description
        lambda_val = st.session_state.lambda_param
        if lambda_val == 0:
            lambda_desc = "Equal Cost"
        elif lambda_val >= 9.5:
            lambda_desc = "Cost per sqm"
        else:
            lambda_desc = f"{(lambda_val/10)*100:.0f}% proportional"
        st.metric("Balance Setting", lambda_desc)
        
        if house.shared_room:
            st.metric("Shared Cost/Person", f"£{house.shared_cost_per_person:.0f}")
        
        st.markdown("**Room Overview:**")
        for room in house.individual_rooms:
            st.write(f"• **{room.room_id}**: {room.size:.1f}m² (Score: {room.desirability_score:.1f}/5)")
        
        # Quick recommendations based on current analysis
        if 'comparison_results' in st.session_state:
            st.markdown("---")
            st.subheader("💡 Quick Tips")
            recommendations = st.session_state.comparison_results.get_recommendations()
            for tip, value in recommendations.items():
                st.caption(f"**{tip}**: {value}")
    
    # Tabs for methodology only
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["ℹ️ Methodology", "📊 Detailed Analysis"])
    
    with tab1:
        render_methodology()
    
    with tab2:
        if 'comparison_results' in st.session_state:
            st.session_state.comparison_results.render_comparison_charts()
        else:
            st.info("Detailed analysis will appear here after running optimization.")


def render_export_modal():
    """Render export options in a modal-like container."""
    if 'house' not in st.session_state or not st.session_state.house:
        st.error("No data to export")
        return
    
    # Create export data
    house = st.session_state.house
    lambda_val = st.session_state.lambda_param
    
    # Run current optimization for export
    result = run_optimization(house, lambda_val)
    
    if not result:
        st.error("Unable to generate export data")
        return
    
    with st.expander("📊 Export Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate summary
            summary_text = f"""# Rent Allocation Summary

**House Details:**
- Total Rent: £{house.total_rent:,.0f}
- Number of People: {house.num_people}
- Shared Room: {house.shared_room.size:.1f} sqm
- Method: Balance = {lambda_val:.1f} ({("Equal Cost" if lambda_val == 0 else "Cost per sqm" if lambda_val >= 9.5 else f"{(lambda_val/10)*100:.0f}% proportional")})

**Key Metrics:**
- Range: £{result.fairness_metrics['Range_Max_Min']:.0f}
- Standard Deviation: £{result.fairness_metrics['Standard_Deviation']:.2f}
- Gini Coefficient: {result.fairness_metrics['Gini_Coefficient']:.3f}

**Rent Allocation:**
"""
            for room in house.individual_rooms:
                total_cost = result.allocation[room.room_id]
                summary_text += f"- {room.room_id} ({room.size:.1f}m²): £{total_cost:.0f}/month\n"
            
            st.text_area("Summary", summary_text, height=300)
        
        with col2:
            st.subheader("📁 Download Options")
            
            # Single CSV download (current results)
            csv_data = []
            for room in house.individual_rooms:
                total_cost = result.allocation[room.room_id]
                individual_cost = total_cost - house.shared_cost_per_person
                
                csv_data.append({
                    'Room_ID': room.room_id,
                    'Size_sqm': room.size,
                    'Size_Score': room.desirability_factors.size_score,
                    'Noise_Level': room.desirability_factors.noise_level,
                    'Accessibility': room.desirability_factors.accessibility,
                    'Desirability_Score': room.desirability_score,
                    'Individual_Cost': round(individual_cost, 2),
                    'Shared_Cost': round(house.shared_cost_per_person, 2),
                    'Total_Monthly_Cost': round(total_cost, 2),
                    'Balance_Value': lambda_val
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Current Results CSV",
                data=csv_string,
                file_name=f"rent_allocation_balance_{lambda_val}.csv",
                mime="text/csv",
                type="secondary"
            )
            
            # Complete package download (comparison + charts)
            if 'comparison_results' in st.session_state:
                st.markdown("---")
                st.markdown("**🎁 Complete Analysis Package**")
                st.markdown("*Includes all data tables + interactive charts*")
                
                if st.button("📦 Prepare Download Package", type="primary"):
                    with st.spinner("Creating export package..."):
                        try:
                            zip_data = st.session_state.comparison_results.create_export_package()
                            
                            if zip_data:
                                st.download_button(
                                    label="💾 Download Complete Package (ZIP)",
                                    data=zip_data,
                                    file_name="rent_analysis_complete.zip",
                                    mime="application/zip",
                                    type="primary"
                                )
                                
                                st.success("✅ Package includes:")
                                st.markdown("""
                                - `detailed_results.csv` - All balance settings & room costs
                                - `fairness_metrics.csv` - Fairness analysis across settings  
                                - `cost_summary.csv` - Cost comparison table
                                - Interactive charts as HTML files
                                """)
                            else:
                                st.error("Unable to create export package")
                        except Exception as e:
                            st.error(f"Error creating package: {str(e)}")
            else:
                st.info("Run analysis first to get complete package option")


def render_methodology():
    """Render methodology information."""
    st.markdown("""
    ### Methodology: Quadratic Optimization for Fair Rent Allocation
    
    This tool uses **quadratic optimization** to find the optimal balance between equality and proportionality in rent allocation.
    
    #### 🎯 **The Optimization Problem**
    
    We minimize: **Σ(cost_i - equal_split)² + λ × Σ(cost_i - proportional_i)²**
    
    Where:
    - **cost_i** = monthly rent for room i
    - **equal_split** = total_rent ÷ number_of_people  
    - **proportional_i** = rent based on room value (size × desirability)
    - **λ (balance parameter)** = weighting factor that controls the balance
    
    #### ⚖️ **How the Balance Parameter Works**
    
    - **Balance = 0**: Pure equality → everyone pays exactly the same amount
    - **Balance = 5**: Balanced approach → considers both equality and room differences  
    - **Balance = 10**: Proportional → rent purely based on room value
    
    #### 📏 **Room Desirability Calculation**
    
    Each room gets a desirability score (1-5) based on:
    - **Size Quality (50% weight)**: How spacious the room feels
    - **Noise Level (40% weight)**: How quiet the room is (1=noisy, 5=quiet)
    - **Accessibility (10% weight)**: Convenience to kitchen/bathroom
    
    **Final room value** = room_size × desirability_score
    
    #### 📊 **Fairness Metrics**
    
    - **Range**: Difference between highest and lowest rent
    - **Standard Deviation**: How much rents vary around the average
    - **Gini Coefficient**: Measure of inequality (0=perfect equality, 1=maximum inequality)
    
    #### 💡 **Practical Guidelines**
    
    **Start with Balance = 5** for a balanced approach. Then adjust:
    - **Move left (lower balance)** if you prefer more equal costs
    - **Move right (higher balance)** if room differences are significant
    - **Use Balance = 0** if you want everyone to pay exactly the same
    
    The optimization automatically ensures the total equals your specified rent while finding the fairest allocation according to your chosen balance point.
    """)


if __name__ == "__main__":
    main()