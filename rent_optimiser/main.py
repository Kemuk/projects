#!/usr/bin/env python3
"""
Intuitive Streamlit app for rent optimization - designed for non-technical housemates.
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
    .rent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .rent-amount {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .room-name {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .method-description {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        text-align: center;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
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
            st.session_state.house = create_fallback_house()
    if 'spread_percent' not in st.session_state:
        st.session_state.spread_percent = 50.0
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
        return None

def get_spread_description(spread_percent: float, house: House) -> str:
    """Get user-friendly description of the current spread setting."""
    if spread_percent <= 5:
        return "🟰 Everyone pays the same amount"
    elif spread_percent >= 95:
        return "📏 Rent based purely on room value"
    else:
        return f"⚖️ {spread_percent:.0f}% consideration for room differences"

def render_sidebar() -> House:
    """Render the simplified sidebar with house configuration only."""
    # Validate house object first
    if not st.session_state.house or not hasattr(st.session_state.house, 'total_rent'):
        st.sidebar.error("⚠️ House data not loaded properly")
        if st.sidebar.button("🔄 Reload House Data"):
            st.session_state.house = create_fallback_house()
            st.rerun()
        return None
    
    # House configuration
    st.sidebar.header("🏠 House Configuration")
    
    # Quick load defaults
    if st.sidebar.button("📋 Load Default House Data", type="secondary", help="Load your house with default settings"):
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
    
    # Room configuration
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
                "Size",
                options=np.arange(0, 5.5, 0.5),
                value=room.desirability_factors.size_score,
                key=f"feel_{i}",
                help="How spacious does this room feel? (50% weight)"
            )
        
        with col2:
            noise_level = st.select_slider(
                "Quiet",
                options=np.arange(0, 5.5, 0.5),
                value=room.desirability_factors.noise_level,
                key=f"quiet_{i}",
                help="How quiet is this room? (40% weight)"
            )
        
        with col3:
            accessibility = st.select_slider(
                "Accessibility",
                options=np.arange(0, 5.5, 0.5),
                value=room.desirability_factors.accessibility,
                key=f"access_{i}",
                help="How convenient to kitchen/bathroom? (10% weight)"
            )
        
        # Show current desirability score
        current_factors = DesirabilityFactors(size_score, noise_level, accessibility)
        current_score = current_factors.calculate_overall_score()
        st.sidebar.caption(f"Score: {current_score:.1f}/5")
        st.sidebar.markdown("---")
        
        rooms_data.append({
            'room_id': room.room_id,
            'size': room_size,
            'size_score': float(size_score),
            'noise_level': float(noise_level),
            'accessibility': float(accessibility)
        })
    
    # Auto-update
    auto_update = st.sidebar.checkbox("🔄 Auto-update results", value=True, help="Update results as you change settings")
    
    # Update house when settings change
    should_update = auto_update
    if not auto_update:
        should_update = st.sidebar.button("📊 Update House Settings", type="primary")
    
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

def run_optimization(house: House, spread_percent: float) -> OptimizationResult:
    """Run optimization with spread percentage."""
    try:
        optimizer = QuadraticOptimizer.from_spread_percentage(spread_percent, house)
        result = optimizer.get_optimization_result(house)
        return result
    except Exception as e:
        st.error(f"Error running optimization: {str(e)}")
        return None

def render_room_cards(house: House, result: OptimizationResult):
    """Render room cards with color coding."""
    st.subheader("💰 Monthly Rent for Each Room")
    
    # Get room cards data
    optimizer = QuadraticOptimizer.from_spread_percentage(st.session_state.spread_percent, house)
    cards_data = optimizer.get_room_cards_data(house, result.allocation)
    
    # Sort by cost for better visualization
    cards_data.sort(key=lambda x: x['total_cost'], reverse=True)
    
    # Display in columns
    cols = st.columns(3)
    for i, card_data in enumerate(cards_data):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background-color: {card_data['color']}; padding: 1rem; border-radius: 0.5rem; color: white; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">{card_data['room_id']}</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{card_data['cost_formatted']}</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">{card_data['size_formatted']} • {card_data['deviation_formatted']} vs avg</div>
            </div>
            """, unsafe_allow_html=True)

def render_calculate_rent_tab(house: House):
    """Render the Calculate Rent tab with slider and results."""
    
    # Spread slider
    st.subheader("🎛️ How much should room differences matter?")
    
    spread_percent = st.slider(
        "Rent Spread",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.spread_percent,
        step=1.0,
        help="0% = Everyone pays the same | 100% = Rent proportional to room value"
    )
    
    # Update session state
    st.session_state.spread_percent = spread_percent
    
    # Show current method description
    description = get_spread_description(spread_percent, house)
    st.markdown(f'<div class="method-description">{description}</div>', unsafe_allow_html=True)
    
    # Show practical lambda range info
    practical_lambda = house.get_practical_lambda_range()
    current_lambda = house.spread_to_lambda_for_house(spread_percent)
    
    with st.expander("ℹ️ Technical Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current λ", f"{current_lambda:.2f}")
        with col2:
            st.metric("Max Effective λ", f"{practical_lambda:.2f}")
        with col3:
            efficiency = (current_lambda / practical_lambda) * 100 if practical_lambda > 0 else 0
            st.metric("Range Efficiency", f"{efficiency:.0f}%")
        
        st.caption(f"This house's practical lambda range is 0 to {practical_lambda:.2f}. "
                   f"Beyond {practical_lambda:.2f}, changes in rent allocation become negligible (< £1 per room).")
    
    # Quick preset buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Equal Split", help="Everyone pays the same"):
            st.session_state.spread_percent = 0.0
            st.rerun()
    
    with col2:
        if st.button("Balanced", help="Consider room differences moderately"):
            st.session_state.spread_percent = 50.0
            st.rerun()
    
    with col3:
        if st.button("Proportional", help="Rent based on room value"):
            st.session_state.spread_percent = 100.0
            st.rerun()
    
    st.markdown("---")
    
    # Run optimization
    result = run_optimization(house, spread_percent)
    
    if result:
        # Room cards
        render_room_cards(house, result)
        
        # Summary stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        costs = [result.allocation[room.room_id] for room in house.individual_rooms]
        
        with col1:
            st.metric("Cheapest Room", f"£{min(costs):.0f}")
        
        with col2:
            st.metric("Most Expensive", f"£{max(costs):.0f}")
        
        with col3:
            st.metric("Difference", f"£{max(costs) - min(costs):.0f}")
        
        with col4:
            fairness_score = 1 - result.fairness_metrics.get('Gini_Coefficient', 0)
            st.metric("Fairness Score", f"{fairness_score:.0%}", help="Higher = more fair distribution")
        
        # Results table
        st.subheader("📊 Detailed Breakdown")
        
        # Get table data
        optimizer = QuadraticOptimizer.from_spread_percentage(spread_percent, house)
        table_data = optimizer.get_display_table_data(house, result.allocation)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Bar chart
        st.subheader("📈 Visual Breakdown")
        
        plot_data = []
        for room in house.individual_rooms:
            total_cost = result.allocation[room.room_id]
            deviation = total_cost - house.target_mean
            
            plot_data.append({
                'Room': room.room_id,
                'Total_Cost': total_cost,
                'Deviation': deviation,
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        fig_bar = px.bar(
            df_plot,
            x='Room',
            y='Total_Cost',
            title='Monthly Rent by Room',
            color='Deviation',
            color_continuous_scale='RdYlGn_r',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Deviation': 'vs Average (£)'}
        )
        
        # Add average line
        fig_bar.add_hline(
            y=house.target_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text="Equal Split"
        )
        
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.error("Unable to calculate results. Please check your settings.")

def render_compare_methods_tab(house: House):
    """Render the Compare Methods tab with custom range builder."""
    st.subheader("🔍 Compare Different Methods")
    
    # Show practical range information
    practical_lambda = house.get_practical_lambda_range()
    st.info(f"📊 **Your house's practical spread range:** 0% to 100% (λ: 0 to {practical_lambda:.2f})")
    
    # Initialize comparison
    comparison = OptimizationComparison(house)
    
    # Custom range builder
    comparison.render_custom_range_builder()
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    try:
        recommendations = comparison.get_spread_recommendations()
        if recommendations:
            cols = st.columns(len(recommendations))
            for i, (criteria, recommendation) in enumerate(recommendations.items()):
                with cols[i]:
                    st.metric(criteria, recommendation)
    except Exception as e:
        st.info("Run comparisons above to see recommendations")

def render_room_details_tab(house: House):
    """Render the Room Details tab."""
    st.subheader("🏠 Room Information")
    
    # Room details table
    room_data = []
    for room in house.individual_rooms:
        room_data.append({
            'Room': room.room_id,
            'Size (m²)': f"{room.size:.1f}",
            'Spaciousness': f"{room.desirability_factors.size_score:.1f}/5",
            'Quietness': f"{room.desirability_factors.noise_level:.1f}/5",
            'Accessibility': f"{room.desirability_factors.accessibility:.1f}/5",
            'Overall Score': f"{room.desirability_score:.1f}/5"
        })
    
    df_rooms = pd.DataFrame(room_data)
    st.dataframe(df_rooms, use_container_width=True, hide_index=True)
    
    # House summary
    st.subheader("📋 House Summary")
    
    summary = house.get_house_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rent", f"£{summary['total_rent']:,.0f}")
        st.metric("People", summary['num_people'])
    
    with col2:
        st.metric("Average Rent", f"£{summary['average_rent']:.0f}")
        st.metric("Shared Cost/Person", f"£{summary['shared_cost_per_person']:.0f}")
    
    with col3:
        st.metric("Total Area", f"{summary['total_area']:.1f}m²")
        st.metric("Shared Room", f"{summary['shared_room_size']:.1f}m²")
    
    # Explanation
    st.markdown("---")
    st.subheader("🤔 How Room Scoring Works")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>Room Quality Calculation:</strong><br>
    • <strong>Spaciousness (50% weight):</strong> How big the room feels<br>
    • <strong>Quietness (40% weight):</strong> How peaceful the room is<br>
    • <strong>Accessibility (10% weight):</strong> How convenient to kitchen/bathroom<br><br>
    <strong>Room Value = Size × Quality Score</strong><br>
    This room value is used to determine proportional rent allocation.
    </div>
    """, unsafe_allow_html=True)

def render_export_modal():
    """Render export options."""
    if 'house' not in st.session_state or not st.session_state.house:
        st.error("No data to export")
        return
    
    # Create export data
    house = st.session_state.house
    spread_percent = st.session_state.spread_percent
    
    # Run current optimization for export
    result = run_optimization(house, spread_percent)
    
    if not result:
        st.error("Unable to generate export data")
        return
    
    with st.expander("📊 Export Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate summary
            practical_lambda = house.get_practical_lambda_range()
            current_lambda = house.spread_to_lambda_for_house(spread_percent)
            
            summary_text = f"""# Rent Allocation Summary

**House Details:**
- Total Rent: £{house.total_rent:,.0f}
- Number of People: {house.num_people}
- Shared Room: {house.shared_room.size:.1f} sqm
- Method: {get_spread_description(spread_percent, house)}
- Spread: {spread_percent:.1f}% (λ={current_lambda:.2f}, max λ={practical_lambda:.2f})

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
            
            # Single CSV download
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
                    'Spread_Percent': spread_percent
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Current Results CSV",
                data=csv_string,
                file_name=f"rent_allocation_spread_{spread_percent:.0f}percent.csv",
                mime="text/csv",
                type="secondary"
            )

def main():
    """Main Streamlit app function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    col_title, col_export = st.columns([4, 1])
    
    with col_title:
        st.markdown('<p class="main-header">🏠 Fair Rent Calculator</p>', unsafe_allow_html=True)
        st.markdown("**Find the fair rent for each room in your house**")
    
    with col_export:
        st.markdown("### ")  # Space for alignment
        if st.button("💾 Export Results", type="secondary", help="Export current results"):
            render_export_modal()
    
    # Sidebar
    house = render_sidebar()
    
    # Check if house is valid
    if not house:
        st.error("🏠 Unable to load house configuration. Please check the sidebar to reload.")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💰 Calculate Rent", "🔍 Compare Methods", "🏠 Room Details"])
    
    with tab1:
        render_calculate_rent_tab(house)
    
    with tab2:
        render_compare_methods_tab(house)
    
    with tab3:
        render_room_details_tab(house)

if __name__ == "__main__":
    main()