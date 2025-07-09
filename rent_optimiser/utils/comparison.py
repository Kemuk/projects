import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from typing import List, Dict, Optional, Tuple
import zipfile
import io
from optimizers.quadratic_optimizer import QuadraticOptimizer
from optimizers.base_optimizer import OptimizationResult


class OptimizationComparison:
    """
    Streamlit-adapted optimization comparison utility.
    
    Handles running multiple lambda values, creating interactive comparisons,
    and exporting results with charts in a Streamlit-friendly way.
    """
    
    def __init__(self, house):
        """Initialize with a house object."""
        self.house = house
        self.results = []
        self.comparison_df = None
        self.fairness_df = None
        self.current_charts = {}
        self.custom_range_data = None
    
    def run_lambda_comparison(self, lambda_values: List[float] = None) -> Dict:
        """
        Run comparison across different lambda values.
        
        Args:
            lambda_values: List of lambda values to test
            
        Returns:
            Dictionary with comparison results
        """
        if lambda_values is None:
            lambda_values = [0.0, 1.0, 2.0, 5.0, 7.0, 10.0]
        
        self.results = []
        
        # Run optimization for each lambda
        for lambda_val in lambda_values:
            optimizer = QuadraticOptimizer(lambda_val)
            result = optimizer.get_optimization_result(self.house)
            self.results.append(result)
        
        # Create comparison dataframes
        self._create_comparison_dataframes()
        
        return {
            'detailed_results': self.comparison_df,
            'fairness_metrics': self.fairness_df,
            'lambda_values': lambda_values
        }
    
    def run_custom_range_comparison(self, spread_percentages: List[float]) -> pd.DataFrame:
        """
        Run comparison for user-selected spread percentages.
        
        Args:
            spread_percentages: List of spread percentages (0-100)
            
        Returns:
            DataFrame with comparison data
        """
        comparison_data = []
        
        for spread_percent in spread_percentages:
            optimizer = QuadraticOptimizer.from_spread_percentage(spread_percent)
            result = optimizer.get_optimization_result(self.house)
            
            for room in self.house.individual_rooms:
                cost = result.allocation[room.room_id]
                comparison_data.append({
                    'Spread': f"{spread_percent:.0f}%",
                    'Spread_Value': spread_percent,
                    'Room': room.room_id,
                    'Cost': cost,
                    'Size': room.size,
                    'Desirability': room.desirability_score,
                    'Cost_per_sqm': cost / room.size,
                    'Deviation_from_Mean': cost - self.house.target_mean
                })
        
        self.custom_range_data = pd.DataFrame(comparison_data)
        return self.custom_range_data
    
    def get_trajectory_data(self, num_points: int = 20) -> pd.DataFrame:
        """
        Generate trajectory data for rent changes across spread percentages.
        
        Args:
            num_points: Number of points to generate between 0-100%
            
        Returns:
            DataFrame with trajectory data
        """
        spread_percentages = np.linspace(0, 100, num_points)
        return self.run_custom_range_comparison(spread_percentages)
    
    def render_custom_range_builder(self):
        """Render the custom range builder interface."""
        st.subheader("🎯 Custom Range Builder")
        st.markdown("*Select specific spread percentages to compare*")
        
        # Default percentages
        default_percentages = [0, 25, 50, 75, 100]
        
        # Multi-select for percentages
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_percentages = st.multiselect(
                "Select spread percentages to compare:",
                options=list(range(0, 101, 5)),  # 0, 5, 10, 15, ..., 100
                default=default_percentages,
                help="Choose multiple percentages to see how rent changes"
            )
        
        with col2:
            # Quick add buttons
            st.markdown("**Quick Add:**")
            if st.button("Add Current Setting"):
                current_spread = st.session_state.get('spread_percent', 50.0)
                if current_spread not in selected_percentages:
                    selected_percentages.append(current_spread)
                    st.rerun()
            
            if st.button("Add Common Values"):
                common_values = [0, 20, 40, 60, 80, 100]
                for val in common_values:
                    if val not in selected_percentages:
                        selected_percentages.append(val)
                st.rerun()
        
        if not selected_percentages:
            st.warning("Please select at least one percentage to compare.")
            return
        
        # Sort percentages
        selected_percentages.sort()
        
        # Run comparison
        with st.spinner("Calculating comparisons..."):
            comparison_df = self.run_custom_range_comparison(selected_percentages)
        
        # Display results
        self._render_custom_range_results(comparison_df, selected_percentages)
    
    def _render_custom_range_results(self, df: pd.DataFrame, percentages: List[float]):
        """Render the custom range comparison results."""
        
        # Create pivot table for easy comparison
        st.subheader("📊 Range Comparison Table")
        
        pivot_df = df.pivot(index='Room', columns='Spread', values='Cost')
        
        # Add room info
        room_info = df[['Room', 'Size', 'Desirability']].drop_duplicates().set_index('Room')
        display_df = pd.concat([room_info, pivot_df], axis=1)
        
        # Format costs as currency
        currency_columns = [col for col in display_df.columns if '%' in str(col)]
        
        styled_df = display_df.style.format({
            'Size': '{:.1f}m²',
            'Desirability': '{:.1f}',
            **{col: '£{:.0f}' for col in currency_columns}
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            fig_bars = px.bar(
                df,
                x='Room',
                y='Cost',
                color='Spread',
                barmode='group',
                title='Rent Comparison Across Spreads',
                labels={'Cost': 'Monthly Rent (£)'}
            )
            
            # Add average line
            fig_bars.add_hline(
                y=self.house.target_mean,
                line_dash="dash",
                line_color="gray",
                annotation_text="Equal Split"
            )
            
            st.plotly_chart(fig_bars, use_container_width=True)
            self.current_charts['custom_bars'] = fig_bars
        
        with col2:
            # Cost per sqm comparison
            fig_per_sqm = px.bar(
                df,
                x='Room',
                y='Cost_per_sqm',
                color='Spread',
                barmode='group',
                title='Cost per m² Comparison',
                labels={'Cost_per_sqm': 'Cost per m² (£)'}
            )
            
            st.plotly_chart(fig_per_sqm, use_container_width=True)
            self.current_charts['custom_per_sqm'] = fig_per_sqm
        
        # Trajectory chart
        st.subheader("📈 Rent Trajectories")
        
        # Generate smooth trajectory data
        trajectory_df = self.get_trajectory_data(num_points=21)  # 0, 5, 10, ..., 100
        
        fig_trajectory = px.line(
            trajectory_df,
            x='Spread_Value',
            y='Cost',
            color='Room',
            title='How Rent Changes with Spread Percentage',
            labels={'Spread_Value': 'Spread Percentage (%)', 'Cost': 'Monthly Rent (£)'},
            markers=True
        )
        
        # Add current setting indicator
        current_spread = st.session_state.get('spread_percent', 50.0)
        fig_trajectory.add_vline(
            x=current_spread,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Setting ({current_spread}%)"
        )
        
        # Add equal split line
        fig_trajectory.add_hline(
            y=self.house.target_mean,
            line_dash="dot",
            line_color="gray",
            annotation_text="Equal Split"
        )
        
        st.plotly_chart(fig_trajectory, use_container_width=True)
        self.current_charts['custom_trajectory'] = fig_trajectory
        
        # Summary metrics
        st.subheader("📋 Summary Metrics")
        
        summary_data = []
        for spread_percent in percentages:
            spread_data = df[df['Spread_Value'] == spread_percent]
            costs = spread_data['Cost'].values
            
            summary_data.append({
                'Spread': f"{spread_percent:.0f}%",
                'Min Cost': f"£{min(costs):.0f}",
                'Max Cost': f"£{max(costs):.0f}",
                'Range': f"£{max(costs) - min(costs):.0f}",
                'Std Dev': f"£{np.std(costs):.0f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    def get_spread_recommendations(self) -> Dict[str, str]:
        """Get recommendations based on spread percentage analysis."""
        if self.custom_range_data is None:
            # Generate trajectory data for analysis
            self.get_trajectory_data(num_points=21)
        
        recommendations = {}
        
        # Calculate metrics for each spread percentage
        spread_metrics = []
        for spread_percent in self.custom_range_data['Spread_Value'].unique():
            spread_data = self.custom_range_data[self.custom_range_data['Spread_Value'] == spread_percent]
            costs = spread_data['Cost'].values
            
            spread_metrics.append({
                'spread': spread_percent,
                'range': max(costs) - min(costs),
                'std_dev': np.std(costs),
                'gini': self._calculate_gini(costs)
            })
        
        metrics_df = pd.DataFrame(spread_metrics)
        
        # Find best options
        best_equality = metrics_df.loc[metrics_df['gini'].idxmin(), 'spread']
        best_range = metrics_df.loc[metrics_df['range'].idxmin(), 'spread']
        best_stability = metrics_df.loc[metrics_df['std_dev'].idxmin(), 'spread']
        
        recommendations['Most Equal Distribution'] = f"{best_equality:.0f}%"
        recommendations['Smallest Cost Range'] = f"{best_range:.0f}%"
        recommendations['Most Stable Costs'] = f"{best_stability:.0f}%"
        
        return recommendations
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values."""
        if len(values) <= 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumulative_values = np.cumsum(sorted_values)
        
        if cumulative_values[-1] == 0:
            return 0.0
            
        return (n + 1 - 2 * sum(cumulative_values) / cumulative_values[-1]) / n
    
    def _create_comparison_dataframes(self):
        """Create comparison dataframes from results."""
        # Detailed results comparison
        all_data = []
        fairness_data = []
        
        for result in self.results:
            lambda_val = result.parameters['lambda_param']
            
            # Add fairness metrics
            fairness_row = result.fairness_metrics.copy()
            fairness_row['Lambda'] = lambda_val
            fairness_row['Description'] = QuadraticOptimizer(lambda_val).get_lambda_description()
            fairness_data.append(fairness_row)
            
            # Add detailed allocation data
            for room in self.house.individual_rooms:
                total_cost = result.allocation[room.room_id]
                individual_cost = total_cost - self.house.shared_cost_per_person
                
                all_data.append({
                    'Lambda': lambda_val,
                    'Description': QuadraticOptimizer(lambda_val).get_lambda_description(),
                    'Room': room.room_id,
                    'Size_sqm': room.size,
                    'Desirability': room.desirability_score,
                    'Total_Cost': total_cost,
                    'Individual_Cost': individual_cost,
                    'Shared_Cost': self.house.shared_cost_per_person,
                    'Cost_per_sqm': total_cost / room.size,
                    'Deviation_from_Mean': total_cost - self.house.target_mean
                })
        
        self.comparison_df = pd.DataFrame(all_data)
        self.fairness_df = pd.DataFrame(fairness_data)
    
    def render_fairness_tab(self):
        """Render fairness analysis tab with metrics and charts."""
        st.markdown("### 📊 Fairness Metrics Analysis")
        st.markdown("*Compare fairness across different spreads*")
        
        if self.fairness_df is None:
            st.warning("Run comparison first")
            return
        
        # Convert lambda to spread percentage for display
        fairness_display = self.fairness_df.copy()
        fairness_display['Spread'] = fairness_display['Lambda'].apply(
            lambda x: f"{QuadraticOptimizer.lambda_to_spread(x):.0f}%"
        )
        
        # Fairness metrics table
        st.subheader("📋 Fairness Metrics Table")
        display_cols = ['Spread', 'Description', 'Range_Max_Min', 'Standard_Deviation', 'Gini_Coefficient']
        table_data = fairness_display[display_cols].round(2)
        table_data.columns = ['Spread', 'Description', 'Range (£)', 'Std Deviation (£)', 'Gini Coefficient']
        st.dataframe(table_data, use_container_width=True, hide_index=True)
        
        # Fairness charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Range chart
            fig_range = px.bar(
                fairness_display,
                x='Spread',
                y='Range_Max_Min',
                title='Range vs Spread',
                labels={'Range_Max_Min': 'Range (£)', 'Spread': 'Spread'},
                color='Range_Max_Min',
                color_continuous_scale='Reds'
            )
            fig_range.update_layout(showlegend=False)
            st.plotly_chart(fig_range, use_container_width=True)
            self.current_charts['fairness_range'] = fig_range
        
        with col2:
            # Gini coefficient chart
            fig_gini = px.bar(
                fairness_display,
                x='Spread',
                y='Gini_Coefficient',
                title='Gini Coefficient vs Spread',
                labels={'Gini_Coefficient': 'Gini Coefficient', 'Spread': 'Spread'},
                color='Gini_Coefficient',
                color_continuous_scale='Blues'
            )
            fig_gini.update_layout(showlegend=False)
            st.plotly_chart(fig_gini, use_container_width=True)
            self.current_charts['fairness_gini'] = fig_gini
        
        # Standard deviation chart
        fig_std = px.line(
            fairness_display,
            x='Spread',
            y='Standard_Deviation',
            title='Standard Deviation vs Spread',
            markers=True,
            labels={'Standard_Deviation': 'Standard Deviation (£)', 'Spread': 'Spread'}
        )
        st.plotly_chart(fig_std, use_container_width=True)
        self.current_charts['fairness_std'] = fig_std
        
        # Current setting indicator
        current_spread = st.session_state.get('spread_percent', 50.0)
        current_lambda = QuadraticOptimizer.spread_to_lambda(current_spread)
        
        # Find closest lambda value in comparison data
        lambda_values = self.fairness_df['Lambda'].values
        closest_lambda = lambda_values[np.argmin(np.abs(lambda_values - current_lambda))]
        current_metrics = self.fairness_df[self.fairness_df['Lambda'] == closest_lambda]
        
        if not current_metrics.empty:
            closest_spread = QuadraticOptimizer.lambda_to_spread(closest_lambda)
            if abs(current_spread - closest_spread) > 1:
                st.info(f"**Current Setting ({current_spread:.0f}%, approximated from {closest_spread:.0f}%)**: "
                       f"Range: £{current_metrics['Range_Max_Min'].iloc[0]:.0f}, "
                       f"Gini: {current_metrics['Gini_Coefficient'].iloc[0]:.3f}")
            else:
                st.info(f"**Current Setting ({current_spread:.0f}%)**: "
                       f"Range: £{current_metrics['Range_Max_Min'].iloc[0]:.0f}, "
                       f"Gini: {current_metrics['Gini_Coefficient'].iloc[0]:.3f}")
    
    def render_summary_tab(self):
        """Render cost summary tab with allocation table and charts."""
        st.markdown("### 📋 Cost Allocation Summary")
        st.markdown("*Compare room costs across different spreads*")
        
        if self.comparison_df is None:
            st.warning("Run comparison first")
            return
        
        # Convert lambda to spread percentage for display
        summary_display = self.comparison_df.copy()
        summary_display['Spread'] = summary_display['Lambda'].apply(
            lambda x: f"{QuadraticOptimizer.lambda_to_spread(x):.0f}%"
        )
        
        # Create cost summary pivot table
        st.subheader("💰 Cost Summary Table")
        cost_pivot = summary_display.pivot(
            index='Room',
            columns='Spread',
            values='Total_Cost'
        ).round(0)
        
        # Add room metadata
        room_metadata = summary_display[['Room', 'Size_sqm', 'Desirability']].drop_duplicates()
        room_metadata = room_metadata.set_index('Room')
        
        summary_table = pd.concat([room_metadata, cost_pivot], axis=1)
        st.dataframe(summary_table, use_container_width=True)
        
        # Cost comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Current spread costs
            current_spread = st.session_state.get('spread_percent', 50.0)
            current_lambda = QuadraticOptimizer.spread_to_lambda(current_spread)
            
            # Find closest lambda value in comparison data
            lambda_values = summary_display['Lambda'].unique()
            closest_lambda = lambda_values[np.argmin(np.abs(lambda_values - current_lambda))]
            current_data = summary_display[summary_display['Lambda'] == closest_lambda]
            
            closest_spread = QuadraticOptimizer.lambda_to_spread(closest_lambda)
            title_suffix = f" ({current_spread:.0f}%)" if abs(current_spread - closest_spread) <= 1 else f" (≈{closest_spread:.0f}%)"
            
            fig_current = px.bar(
                current_data,
                x='Room',
                y='Total_Cost',
                title=f'Current Allocation{title_suffix}',
                labels={'Total_Cost': 'Monthly Rent (£)'},
                color='Total_Cost',
                color_continuous_scale='Viridis'
            )
            
            # Add average line
            fig_current.add_hline(
                y=self.house.target_mean,
                line_dash="dash",
                line_color="red",
                annotation_text="Equal Split"
            )
            fig_current.update_layout(showlegend=False)
            st.plotly_chart(fig_current, use_container_width=True)
            self.current_charts['summary_current'] = fig_current
        
        with col2:
            # Cost per square meter comparison
            fig_per_sqm = px.bar(
                current_data,
                x='Room',
                y='Cost_per_sqm',
                title=f'Cost per m²{title_suffix}',
                labels={'Cost_per_sqm': 'Cost per m² (£)'},
                color='Cost_per_sqm',
                color_continuous_scale='Plasma'
            )
            fig_per_sqm.update_layout(showlegend=False)
            st.plotly_chart(fig_per_sqm, use_container_width=True)
            self.current_charts['summary_per_sqm'] = fig_per_sqm
        
        # Range comparison across spreads
        st.subheader("📈 How Costs Change with Spread")
        
        # Line chart showing cost changes
        fig_trends = px.line(
            summary_display,
            x='Lambda',
            y='Total_Cost',
            color='Room',
            title='Room Costs vs Spread',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Lambda': 'Spread (λ)'},
            markers=True
        )
        
        # Add current lambda position
        fig_trends.add_vline(
            x=closest_lambda,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Setting ({current_spread:.0f}%)"
        )
        
        # Add equal split line
        fig_trends.add_hline(
            y=self.house.target_mean,
            line_dash="dot",
            line_color="gray",
            annotation_text="Equal Split"
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        self.current_charts['summary_trends'] = fig_trends
    
    def export_custom_range_data(self, spread_percentages: List[float], format: str = 'csv') -> bytes:
        """Export custom range comparison data."""
        df = self.run_custom_range_comparison(spread_percentages)
        
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format == 'excel':
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_all_export_data(self) -> Dict:
        """Get all data formatted for export including charts."""
        export_data = {}
        
        if self.comparison_df is not None:
            export_data['detailed_csv'] = self.comparison_df
            export_data['fairness_csv'] = self.fairness_df
            export_data['summary_csv'] = self._get_summary_table()
        
        if self.custom_range_data is not None:
            export_data['custom_range_csv'] = self.custom_range_data
        
        export_data['charts'] = self.current_charts
        
        return export_data
    
    def _get_summary_table(self) -> pd.DataFrame:
        """Create summary table for export."""
        cost_pivot = self.comparison_df.pivot(
            index='Room',
            columns='Lambda',
            values='Total_Cost'
        ).round(0)
        
        room_metadata = self.comparison_df[['Room', 'Size_sqm', 'Desirability']].drop_duplicates()
        room_metadata = room_metadata.set_index('Room')
        
        return pd.concat([room_metadata, cost_pivot], axis=1)
    
    def create_export_package(self) -> bytes:
        """Create a zip package with all export data."""
        export_data = self.get_all_export_data()
        
        if not export_data:
            return None
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add CSV files
            if 'detailed_csv' in export_data:
                csv_data = export_data['detailed_csv'].to_csv(index=False)
                zip_file.writestr('detailed_results.csv', csv_data)
            
            if 'fairness_csv' in export_data:
                csv_data = export_data['fairness_csv'].to_csv(index=False)
                zip_file.writestr('fairness_metrics.csv', csv_data)
            
            if 'summary_csv' in export_data:
                csv_data = export_data['summary_csv'].to_csv()
                zip_file.writestr('cost_summary.csv', csv_data)
            
            if 'custom_range_csv' in export_data:
                csv_data = export_data['custom_range_csv'].to_csv(index=False)
                zip_file.writestr('custom_range_comparison.csv', csv_data)
            
            # Add charts as HTML files
            charts = export_data.get('charts', {})
            for chart_name, fig in charts.items():
                html_content = pio.to_html(fig, include_plotlyjs='cdn')
                zip_file.writestr(f'{chart_name}_chart.html', html_content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def get_recommendations(self) -> Dict[str, str]:
        """Get recommendations based on fairness metrics."""
        if self.fairness_df is None:
            return {}
        
        recommendations = {}
        
        # Find best lambda for different criteria
        best_gini_idx = self.fairness_df['Gini_Coefficient'].idxmin()
        best_range_idx = self.fairness_df['Range_Max_Min'].idxmin()
        best_std_idx = self.fairness_df['Standard_Deviation'].idxmin()
        
        # Convert to spread percentages
        best_gini_spread = QuadraticOptimizer.lambda_to_spread(self.fairness_df.loc[best_gini_idx, 'Lambda'])
        best_range_spread = QuadraticOptimizer.lambda_to_spread(self.fairness_df.loc[best_range_idx, 'Lambda'])
        best_std_spread = QuadraticOptimizer.lambda_to_spread(self.fairness_df.loc[best_std_idx, 'Lambda'])
        
        recommendations['Most Equal (Gini)'] = f"{best_gini_spread:.0f}%"
        recommendations['Smallest Range'] = f"{best_range_spread:.0f}%"
        recommendations['Most Stable'] = f"{best_std_spread:.0f}%"
        
        return recommendations