import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from typing import List, Dict, Optional
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
        st.markdown("*Compare fairness across different balance settings*")
        
        if self.fairness_df is None:
            st.warning("Run comparison first")
            return
        
        # Fairness metrics table
        st.subheader("📋 Fairness Metrics Table")
        fairness_display = self.fairness_df[['Lambda', 'Description', 'Range_Max_Min', 
                                           'Standard_Deviation', 'Gini_Coefficient']].round(2)
        fairness_display.columns = ['Balance', 'Description', 'Range (£)', 'Std Deviation (£)', 'Gini Coefficient']
        st.dataframe(fairness_display, use_container_width=True, hide_index=True)
        
        # Fairness charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Range chart
            fig_range = px.bar(
                self.fairness_df,
                x='Lambda',
                y='Range_Max_Min',
                title='Range vs Balance Setting',
                labels={'Range_Max_Min': 'Range (£)', 'Lambda': 'Balance Setting'},
                color='Range_Max_Min',
                color_continuous_scale='Reds'
            )
            fig_range.update_layout(showlegend=False)
            st.plotly_chart(fig_range, use_container_width=True)
            self.current_charts['fairness_range'] = fig_range
        
        with col2:
            # Gini coefficient chart
            fig_gini = px.bar(
                self.fairness_df,
                x='Lambda',
                y='Gini_Coefficient',
                title='Gini Coefficient vs Balance Setting',
                labels={'Gini_Coefficient': 'Gini Coefficient', 'Lambda': 'Balance Setting'},
                color='Gini_Coefficient',
                color_continuous_scale='Blues'
            )
            fig_gini.update_layout(showlegend=False)
            st.plotly_chart(fig_gini, use_container_width=True)
            self.current_charts['fairness_gini'] = fig_gini
        
        # Standard deviation chart
        fig_std = px.line(
            self.fairness_df,
            x='Lambda',
            y='Standard_Deviation',
            title='Standard Deviation vs Balance Setting',
            markers=True,
            labels={'Standard_Deviation': 'Standard Deviation (£)', 'Lambda': 'Balance Setting'}
        )
        st.plotly_chart(fig_std, use_container_width=True)
        self.current_charts['fairness_std'] = fig_std
        
        # Current setting indicator
        current_lambda = st.session_state.get('lambda_param', 5.0)
        # Find closest lambda value in comparison data
        lambda_values = self.fairness_df['Lambda'].values
        closest_lambda = lambda_values[np.argmin(np.abs(lambda_values - current_lambda))]
        current_metrics = self.fairness_df[self.fairness_df['Lambda'] == closest_lambda]
        
        if not current_metrics.empty:
            if current_lambda != closest_lambda:
                st.info(f"**Current Setting (Balance={current_lambda}, approximated from {closest_lambda})**: "
                       f"Range: £{current_metrics['Range_Max_Min'].iloc[0]:.0f}, "
                       f"Gini: {current_metrics['Gini_Coefficient'].iloc[0]:.3f}")
            else:
                st.info(f"**Current Setting (Balance={current_lambda})**: "
                       f"Range: £{current_metrics['Range_Max_Min'].iloc[0]:.0f}, "
                       f"Gini: {current_metrics['Gini_Coefficient'].iloc[0]:.3f}")
    
    def render_summary_tab(self):
        """Render cost summary tab with allocation table and charts."""
        st.markdown("### 📋 Cost Allocation Summary")
        st.markdown("*Compare room costs across different balance settings*")
        
        if self.comparison_df is None:
            st.warning("Run comparison first")
            return
        
        # Create cost summary pivot table
        st.subheader("💰 Cost Summary Table")
        cost_pivot = self.comparison_df.pivot(
            index='Room',
            columns='Lambda',
            values='Total_Cost'
        ).round(0)
        
        # Add room metadata
        room_metadata = self.comparison_df[['Room', 'Size_sqm', 'Desirability']].drop_duplicates()
        room_metadata = room_metadata.set_index('Room')
        
        summary_table = pd.concat([room_metadata, cost_pivot], axis=1)
        st.dataframe(summary_table, use_container_width=True)
        
        # Cost comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Current lambda costs
            current_lambda = st.session_state.get('lambda_param', 5.0)
            # Find closest lambda value in comparison data
            lambda_values = self.comparison_df['Lambda'].unique()
            closest_lambda = lambda_values[np.argmin(np.abs(lambda_values - current_lambda))]
            current_data = self.comparison_df[self.comparison_df['Lambda'] == closest_lambda]
            
            title_suffix = f" (Balance={current_lambda})" if current_lambda == closest_lambda else f" (Balance≈{closest_lambda})"
            
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
                title=f'Cost per sqm{title_suffix}',
                labels={'Cost_per_sqm': 'Cost per sqm (£)'},
                color='Cost_per_sqm',
                color_continuous_scale='Plasma'
            )
            fig_per_sqm.update_layout(showlegend=False)
            st.plotly_chart(fig_per_sqm, use_container_width=True)
            self.current_charts['summary_per_sqm'] = fig_per_sqm
        
        # Range comparison across lambda values
        st.subheader("📈 How Costs Change with Balance Setting")
        
        # Line chart showing cost changes
        fig_trends = px.line(
            self.comparison_df,
            x='Lambda',
            y='Total_Cost',
            color='Room',
            title='Room Costs vs Balance Setting',
            labels={'Total_Cost': 'Monthly Rent (£)', 'Lambda': 'Balance Setting'},
            markers=True
        )
        
        # Add current lambda position
        fig_trends.add_vline(
            x=closest_lambda,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Setting ({current_lambda})" if current_lambda == closest_lambda else f"Current Setting≈{closest_lambda}"
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
    
    def get_all_export_data(self) -> Dict:
        """Get all data formatted for export including charts."""
        if self.comparison_df is None:
            return {}
        
        export_data = {
            'detailed_csv': self.comparison_df,
            'fairness_csv': self.fairness_df,
            'summary_csv': self._get_summary_table(),
            'charts': self.current_charts
        }
        
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
        
        recommendations['Most Equal (Gini)'] = f"Balance = {self.fairness_df.loc[best_gini_idx, 'Lambda']}"
        recommendations['Smallest Range'] = f"Balance = {self.fairness_df.loc[best_range_idx, 'Lambda']}"
        recommendations['Most Stable'] = f"Balance = {self.fairness_df.loc[best_std_idx, 'Lambda']}"
        
        return recommendations