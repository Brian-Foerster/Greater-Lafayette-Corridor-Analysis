"""
APM Corridor Analysis Interactive Visualizer
===========================================

Streamlit-based interactive tool for exploring Phase 1, 2a, and 2b results.

Features:
- Interactive map with corridor overlays
- Data explorer with filters
- Phase comparison dashboard
- Scenario explorer
- Export functionality

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="APM Corridor Analysis",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        margin: 0.25rem;
    }
    .phase1 { background-color: #ff7f0e; color: white; }
    .phase2a { background-color: #2ca02c; color: white; }
    .phase2b { background-color: #d62728; color: white; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    /* Prevent text cursor on metric and info elements */
    [data-testid="metric-container"] {
        pointer-events: none;
        user-select: none;
    }
    [data-testid="metric-container"] * {
        cursor: default !important;
        user-select: none !important;
    }
    /* Prevent interaction with info/success/error boxes */
    [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stError"], [data-testid="stWarning"] {
        pointer-events: none;
        user-select: none;
    }
    [data-testid="stInfo"] *, [data-testid="stSuccess"] *, [data-testid="stError"] *, [data-testid="stWarning"] * {
        cursor: default !important;
        user-select: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_phase2b_results():
    """Load Phase 2b corridor results"""
    path = Path("data/processed/phase2b_corridor_results.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_phase2b_comparison():
    """Load Phase 2b vs Phase 1 and 2a comparison"""
    path = Path("data/processed/phase2b_vs_phase1_phase2a_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_phase2b_recommendation():
    """Load Phase 2b final recommendation"""
    path = Path("data/processed/phase2b_final_recommendation.json")
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_comparison_data():
    """Load Phase 1 vs 2a vs 2b comparison"""
    path = Path("data/processed/phase2b_vs_phase1_phase2a_comparison.csv")
    if path.exists():
        df = pd.read_csv(path)
        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=['corridor_id'], keep='first')
        return df
    return None

@st.cache_data
def load_corridors_geojson():
    """Load corridor geometries"""
    # Try Phase 2b first, fall back to Phase 2a
    for filename in ['apm_phase2a_corridors.geojson', 'apm_iterative_improved_corridors.geojson']:
        path = Path(f"data/processed/{filename}")
        if path.exists():
            return gpd.read_file(path)
    return None

@st.cache_data
def load_stops_geojson():
    """Load stop locations"""
    for filename in ['apm_phase2a_stops.geojson', 'apm_iterative_improved_stops.geojson']:
        path = Path(f"data/processed/{filename}")
        if path.exists():
            return gpd.read_file(path)
    return None

@st.cache_data
def load_gravity_params():
    """Load gravity model parameters"""
    path = Path("data/processed/gravity_params_calibrated.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_mode_choice_params():
    """Load mode choice parameters"""
    path = Path("data/processed/mode_choice_parameters_phase2b.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_synthetic_persons():
    """Load synthetic persons for population analysis"""
    path = Path("data/processed/synthetic_persons.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_synthetic_households():
    """Load synthetic households for population analysis"""
    path = Path("data/processed/synthetic_households.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_od_matrix():
    """Load OD matrix for trip endpoints"""
    # Try parcel-level flows first (better for visualization)
    path = Path("data/processed/od_parcel_flows.csv")
    if path.exists():
        return pd.read_csv(path)
    
    # Fallback to aggregated OD matrix
    path = Path("data/processed/od_matrix_calibrated_phase2b.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_parcels_for_overlays():
    """Load and aggregate parcel data for map overlays"""
    try:
        # Load synthetic persons to get population by parcel
        persons_df = load_synthetic_persons()
        if persons_df is not None and 'parcel_id' in persons_df.columns:
            pop_by_parcel = persons_df.groupby('parcel_id').size().reset_index(name='population')
            
            # Load employment from gravity model (which has jobs estimates)
            gravity_path = Path("data/processed/parcels_improved_gravity_v2.geojson")
            if gravity_path.exists():
                parcels_gdf = gpd.read_file(gravity_path)
                
                # Normalize parcel ID column name (could be PARCEL_ID, ParcelID, or parcel_id)
                id_col = None
                for col in ['PARCEL_ID', 'ParcelID', 'parcel_id', 'PIN']:
                    if col in parcels_gdf.columns:
                        id_col = col
                        break
                
                if id_col:
                    parcels_gdf = parcels_gdf.rename(columns={id_col: 'parcel_id'})
                    
                    # Merge population data
                    parcels_gdf = parcels_gdf.merge(pop_by_parcel, on='parcel_id', how='left')
                    parcels_gdf['population'] = parcels_gdf['population'].fillna(0)
                
                # Jobs should already be in estimated_jobs column
                if 'estimated_jobs' not in parcels_gdf.columns and 'gravity_weight' in parcels_gdf.columns:
                    parcels_gdf['estimated_jobs'] = parcels_gdf['gravity_weight'] * 100
                
                return parcels_gdf
    except Exception as e:
        st.warning(f"Could not load parcel overlay data: {e}")
    return None

# Utility functions
def create_download_link(df, filename, file_label):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'
    return href

def get_confidence_color(confidence):
    """Get color for confidence level"""
    colors = {
        'VERY HIGH': '#2ca02c',
        'HIGH': '#ff7f0e',
        'MEDIUM-HIGH': '#d62728',
        'MEDIUM': '#9467bd'
    }
    return colors.get(confidence, '#7f7f7f')

def create_corridor_map(corridors_gdf, comparison_df, selected_corridors=None, phase='phase2b', 
                       parcels_gdf=None, od_data=None, overlay_options=None):
    """Create interactive Folium map with corridor overlays and optional population/trip layers"""
    # Calculate center
    if corridors_gdf is not None and len(corridors_gdf) > 0:
        center_lat = corridors_gdf.geometry.centroid.y.mean()
        center_lon = corridors_gdf.geometry.centroid.x.mean()
    else:
        center_lat, center_lon = 40.4237, -86.9212  # Default to Purdue area
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add population density layer if requested
    if overlay_options and overlay_options.get('population') and parcels_gdf is not None:
        try:
            if 'population' in parcels_gdf.columns:
                parcels_with_pop = parcels_gdf[parcels_gdf['population'] > 0].copy()
                
                if len(parcels_with_pop) > 0:
                    # Sample to reduce data size if needed (use top population areas)
                    if len(parcels_with_pop) > 5000:
                        parcels_with_pop = parcels_with_pop.nlargest(5000, 'population')
                    
                    # Create point markers instead of choropleth (more reliable)
                    import math
                    for idx, row in parcels_with_pop.iterrows():
                        if hasattr(row.geometry, 'centroid'):
                            centroid = row.geometry.centroid
                            # Scale radius based on population (uncapped - show actual values)
                            # Use log scale for large populations like Purdue dorms
                            pop = row['population']
                            # Guard against invalid population values
                            if not isinstance(pop, (int, float)) or pd.isna(pop) or pop <= 0:
                                radius = 2
                            elif pop > 1000:
                                # Log scale for very large (e.g., 30k dorm -> ~18 radius)
                                radius = min(5 + math.log10(pop) * 3, 25)
                            else:
                                # Linear for normal populations
                                radius = min(max(pop / 10, 2), 12)
                            folium.CircleMarker(
                                location=[centroid.y, centroid.x],
                                radius=radius,
                                color='red',
                                fill=True,
                                fillColor='red',
                                fillOpacity=0.5,
                                weight=1,
                                popup=f"Population: {row['population']:.0f}<br>Parcel: {row.get('parcel_id', 'Unknown')}",
                                tooltip=f"{row['population']:.0f} people"
                            ).add_to(m)
                    
                    # Add note about data quality
                    st.sidebar.info(f"Showing {len(parcels_with_pop):,} parcels with population")
                else:
                    st.sidebar.warning("No population data available for overlay")
        except Exception as e:
            st.error(f"Population overlay error: {e}")
    
    # Add employment centers layer if requested
    if overlay_options and overlay_options.get('employment') and parcels_gdf is not None:
        try:
            if 'estimated_jobs' in parcels_gdf.columns:
                parcels_with_jobs = parcels_gdf[parcels_gdf['estimated_jobs'] > 5].copy()
                
                if len(parcels_with_jobs) > 0:
                    # Show top employment centers
                    for idx, row in parcels_with_jobs.nlargest(100, 'estimated_jobs').iterrows():
                        if hasattr(row.geometry, 'centroid'):
                            centroid = row.geometry.centroid
                            radius = min(max(row['estimated_jobs'] / 10, 5), 15)
                            folium.CircleMarker(
                                location=[centroid.y, centroid.x],
                                radius=radius,
                                color='purple',
                                fill=True,
                                fillColor='purple',
                                fillOpacity=0.7,
                                weight=2,
                                popup=f"Est. Jobs: {row['estimated_jobs']:.0f}",
                                tooltip=f"Employment: {row['estimated_jobs']:.0f} jobs"
                            ).add_to(m)
        except Exception as e:
            st.error(f"Employment overlay error: {e}")
    
    # Add major trip flows layer if requested
    if overlay_options and overlay_options.get('trips') and od_data is not None and parcels_gdf is not None:
        try:
            # Check if this is parcel-level OD data
            if 'origin_parcel' in od_data.columns and 'dest_parcel' in od_data.columns and 'trips' in od_data.columns:
                # Get top flows
                top_flows = od_data.nlargest(100, 'trips')  # Increased from 50 to 100
                
                # Track success/failure
                flows_drawn = 0
                flows_failed = 0
                
                # Draw lines for major flows
                for idx, flow in top_flows.iterrows():
                    try:
                        origin_id = flow['origin_parcel']
                        dest_id = flow['dest_parcel']
                        
                        # Find parcels by parcel_id column
                        origin_parcel = parcels_gdf[parcels_gdf['parcel_id'] == origin_id]
                        dest_parcel = parcels_gdf[parcels_gdf['parcel_id'] == dest_id]
                        
                        if len(origin_parcel) > 0 and len(dest_parcel) > 0:
                            origin_geom = origin_parcel.iloc[0].geometry.centroid
                            dest_geom = dest_parcel.iloc[0].geometry.centroid
                            
                            # Calculate distance to avoid drawing very short lines
                            dist = ((origin_geom.y - dest_geom.y)**2 + (origin_geom.x - dest_geom.x)**2)**0.5
                            
                            if dist > 0.001:  # Only draw if > ~100m apart
                                folium.PolyLine(
                                    locations=[[origin_geom.y, origin_geom.x], [dest_geom.y, dest_geom.x]],
                                    color='orange',
                                    weight=max(flow['trips'] / 3, 1.5),  # Better visibility
                                    opacity=0.7,
                                    popup=f"<b>{flow['trips']:.0f} trips</b><br>From: {origin_id[:15]}...<br>To: {dest_id[:15]}...",
                                    tooltip=f"{flow['trips']:.0f} trips"
                                ).add_to(m)
                                flows_drawn += 1
                        else:
                            flows_failed += 1
                    except Exception as e:
                        flows_failed += 1
                        continue
                
                # Report results
                if flows_drawn > 0:
                    st.sidebar.success(f"Showing {flows_drawn} trip flows (top {len(top_flows)} OD pairs)")
                else:
                    st.sidebar.warning(f"Could not draw trip flows: {flows_failed} failed to match parcels")
            else:
                st.sidebar.error("Trip flow data has wrong format - expected origin_parcel, dest_parcel, trips columns")
        except Exception as e:
            st.sidebar.error(f"Trip flow overlay error: {e}")
    
    # Add corridors
    if corridors_gdf is not None and comparison_df is not None:
        for idx, row in corridors_gdf.iterrows():
            corridor_id = row.get('corridor_id', f'C{idx+1}')
            
            # Skip if not in selected corridors
            if selected_corridors and corridor_id not in selected_corridors:
                continue
            
            # Get ridership from comparison data
            corridor_data = comparison_df[comparison_df['corridor_id'] == corridor_id]
            if len(corridor_data) == 0:
                continue
            
            corridor_info = corridor_data.iloc[0]
            
            # Get appropriate ridership based on phase (phase is already 'phase1', 'phase2a', or 'phase2b')
            ridership_col = f'{phase}_daily_riders'
            ridership = corridor_info.get(ridership_col, 0)
            
            # Color by ridership
            if ridership > 10000:
                color = '#2ca02c'  # Green - high
            elif ridership > 8000:
                color = '#ff7f0e'  # Orange - medium
            else:
                color = '#d62728'  # Red - low
            
            # Create popup
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0; color: {color};">{corridor_id}</h4>
                <hr style="margin: 5px 0;">
                <b>Ridership:</b> {ridership:,.0f} riders/day<br>
                <b>Confidence:</b> {corridor_info.get('confidence', 'N/A')}<br>
            </div>
            """
            
            # Add corridor to map
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=color: {
                    'color': color,
                    'weight': 4,
                    'opacity': 0.7
                },
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
    
    return m

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🚇 APM Corridor Analysis Interactive Visualizer</div>', 
                unsafe_allow_html=True)
    
    # Load data
    phase2b_df = load_phase2b_results()
    comparison_df = load_comparison_data()
    corridors_gdf = load_corridors_geojson()
    stops_gdf = load_stops_geojson()
    gravity_params = load_gravity_params()
    mode_choice_params = load_mode_choice_params()
    
    # Load additional data for enhanced visualizations
    synthetic_persons = load_synthetic_persons()
    synthetic_households = load_synthetic_households()
    od_matrix = load_od_matrix()
    parcels_enriched = load_parcels_for_overlays()
    
    # Check data availability
    if comparison_df is None:
        st.error("⚠️ Phase 2b comparison data not found. Please ensure phase2b_vs_phase1_phase2a_comparison.csv exists in data/processed/")
        st.stop()
    
    # Verify required columns exist
    required_columns = ['corridor_id', 'phase1_daily_riders', 'phase2a_daily_riders', 
                       'phase2b_daily_riders', 'confidence']
    missing_columns = [col for col in required_columns if col not in comparison_df.columns]
    if missing_columns:
        st.error(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(comparison_df.columns.tolist())}")
        st.stop()
    
    # Debug: Show available columns (only in sidebar when in debug mode)
    # Uncomment for debugging:
    # st.sidebar.text(f"Available columns: {comparison_df.columns.tolist()}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Phase selector
        phase_view = st.radio(
            "Select Phase:",
            ['Phase 2b (VERY HIGH)', 'Phase 2a (HIGH)', 'Phase 1 (MEDIUM-HIGH)'],
            index=0
        )
        # Extract phase key (e.g., 'phase2b', 'phase2a', 'phase1')
        phase_num = phase_view.split()[1].lower()
        phase_key = f'phase{phase_num}'
        
        st.markdown("---")
        
        # Top N corridors
        top_n = st.slider("Show Top N Corridors:", 5, 40, 10, step=5)
        
        st.markdown("---")
        
        # Filter by ridership
        min_ridership = st.number_input("Min Daily Ridership:", 0, 15000, 0, step=500)
        
        st.markdown("---")
        
        # Export section
        st.subheader("📥 Export Data")
        if st.button("Download Current View (CSV)"):
            st.info("Use the download links in the Data Explorer tab")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🗺️ Interactive Map", 
        "📈 Phase Comparison",
        "🔍 Data Explorer",
        "👥 Population & Trips",
        "⚙️ Model Parameters"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Executive Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get top corridor
        top_corridor = comparison_df.nlargest(1, 'phase2b_daily_riders').iloc[0]
        
        with col1:
            st.metric(
                "Top Corridor",
                top_corridor['corridor_id'],
                "Phase 2b"
            )
        
        with col2:
            st.metric(
                "Daily Ridership",
                f"{top_corridor['phase2b_daily_riders']:,.0f}",
                f"{top_corridor['phase2b_vs_phase1_pct']:.1f}% vs Phase 1"
            )
        
        with col3:
            st.metric(
                "Confidence Level",
                top_corridor['confidence'],
                "All components validated"
            )
        
        with col4:
            total_corridors = len(comparison_df)
            st.metric(
                "Corridors Evaluated",
                total_corridors,
                "Phase 1, 2a, 2b"
            )
        
        st.markdown("---")
        
        # Top 10 corridors table
        st.subheader("📊 Top 10 Corridors (Phase 2b)")
        
        top_10 = comparison_df.nlargest(10, 'phase2b_daily_riders')[
            ['corridor_id', 'phase2b_daily_riders', 'phase1_daily_riders', 
             'phase2b_vs_phase1_pct', 'confidence']
        ].copy()
        
        top_10.columns = ['Corridor', 'Phase 2b Riders', 'Phase 1 Riders', 
                          'Change (%)', 'Confidence']
        
        # Format numbers
        top_10['Phase 2b Riders'] = top_10['Phase 2b Riders'].apply(lambda x: f"{x:,.0f}")
        top_10['Phase 1 Riders'] = top_10['Phase 1 Riders'].apply(lambda x: f"{x:,.0f}")
        top_10['Change (%)'] = top_10['Change (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(top_10, use_container_width=True)
        
        # Quick insights
        st.markdown("---")
        st.subheader("🔍 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Why Phase 2b Results Changed:**")
            st.markdown("""
            - **-18%**: Empirical OD (survey-based vs synthetic)
            - **-9%**: Calibrated distance decay (measured vs assumed)
            - **-7%**: Local mode choice (survey vs generic NHTS)
            - **Total: -34.6%** reduction from Phase 1
            """)
        
        with col2:
            st.markdown("**Confidence Progression:**")
            st.markdown("""
            - **Phase 1**: MEDIUM-HIGH (synthetic assumptions)
            - **Phase 2a**: HIGH (improved distribution)
            - **Phase 2b**: VERY HIGH (all components validated)
            - **Uncertainty**: ±5-10% (vs ±15-20% in Phase 1)
            """)
    
    # Tab 2: Interactive Map
    with tab2:
        st.header("🗺️ Interactive Corridor Map")
        
        # Map controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info("💡 Click on corridors to see details. Color indicates ridership: Green (>10k), Orange (8-10k), Red (<8k)")
        
        with col2:
            show_all = st.checkbox("Show All Corridors", value=False)
        
        with col3:
            show_overlays = st.checkbox("Show Overlays", value=False)
        
        # Overlay options (if enabled)
        overlay_options = {}
        if show_overlays:
            st.markdown("**Map Overlays:**")
            overlay_col1, overlay_col2, overlay_col3 = st.columns(3)
            with overlay_col1:
                overlay_options['population'] = st.checkbox("Population Density", value=False)
            with overlay_col2:
                overlay_options['employment'] = st.checkbox("Employment Centers", value=False)
            with overlay_col3:
                overlay_options['trips'] = st.checkbox("Major Trip Flows", value=False)
        
        # Filter corridors for map
        ridership_col = f'{phase_key}_daily_riders'
        if show_all:
            filtered_df = comparison_df
        else:
            filtered_df = comparison_df.nlargest(top_n, ridership_col)
        
        filtered_df = filtered_df[filtered_df[ridership_col] >= min_ridership]
        selected_corridor_ids = filtered_df['corridor_id'].tolist()
        
        # Create and display map
        corridor_map = create_corridor_map(
            corridors_gdf, 
            comparison_df, 
            selected_corridor_ids,
            phase_key,
            parcels_gdf=parcels_enriched,
            od_data=od_matrix,
            overlay_options=overlay_options if show_overlays else None
        )
        
        folium_static(corridor_map, width=1200, height=600)
        
        # Map legend
        st.markdown("---")
        st.markdown("**Legend:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **High Ridership** (>10,000 riders/day)")
        with col2:
            st.markdown("🟠 **Medium Ridership** (8,000-10,000 riders/day)")
        with col3:
            st.markdown("🔴 **Lower Ridership** (<8,000 riders/day)")
        
        if show_overlays:
            st.markdown("---")
            st.markdown("**Overlay Legend:**")
            overlay_legend_col1, overlay_legend_col2, overlay_legend_col3 = st.columns(3)
            with overlay_legend_col1:
                if overlay_options.get('population'):
                    st.markdown("🟡 **Population Density** (Red = High, Yellow = Low)")
            with overlay_legend_col2:
                if overlay_options.get('employment'):
                    st.markdown("🟣 **Employment Centers** (Larger circles = More jobs)")
            with overlay_legend_col3:
                if overlay_options.get('trips'):
                    st.markdown("🟠 **Trip Flows** (Lines connect major OD pairs)")
    
    # Tab 3: Phase Comparison
    with tab3:
        st.header("📈 Phase-by-Phase Comparison")
        
        # Corridor selector for detailed comparison
        all_corridors = sorted(comparison_df['corridor_id'].unique())
        selected_for_comparison = st.multiselect(
            "Select Corridors to Compare:",
            all_corridors,
            default=all_corridors[:5]
        )
        
        if len(selected_for_comparison) > 0:
            # Filter data
            comp_data = comparison_df[comparison_df['corridor_id'].isin(selected_for_comparison)]
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Phase 1',
                x=comp_data['corridor_id'],
                y=comp_data['phase1_daily_riders'],
                marker_color='#ff7f0e'
            ))
            
            fig.add_trace(go.Bar(
                name='Phase 2a',
                x=comp_data['corridor_id'],
                y=comp_data['phase2a_daily_riders'],
                marker_color='#2ca02c'
            ))
            
            fig.add_trace(go.Bar(
                name='Phase 2b',
                x=comp_data['corridor_id'],
                y=comp_data['phase2b_daily_riders'],
                marker_color='#d62728'
            ))
            
            fig.update_layout(
                title="Daily Ridership by Phase",
                xaxis_title="Corridor",
                yaxis_title="Daily Riders",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Percentage change analysis
            st.subheader("📉 Percentage Change Analysis")
            
            change_data = comp_data[['corridor_id', 'phase2b_vs_phase1_pct', 'phase2b_vs_phase2a_pct']].copy()
            change_data.columns = ['Corridor', 'Phase 2b vs Phase 1 (%)', 'Phase 2b vs Phase 2a (%)']
            
            fig2 = px.bar(
                change_data,
                x='Corridor',
                y=['Phase 2b vs Phase 1 (%)', 'Phase 2b vs Phase 2a (%)'],
                title="Percentage Change from Previous Phases",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_phase1 = comp_data['phase1_daily_riders'].mean()
                st.metric("Avg Ridership (Phase 1)", f"{avg_phase1:,.0f}")
            
            with col2:
                avg_phase2a = comp_data['phase2a_daily_riders'].mean()
                st.metric("Avg Ridership (Phase 2a)", f"{avg_phase2a:,.0f}")
            
            with col3:
                avg_phase2b = comp_data['phase2b_daily_riders'].mean()
                change = ((avg_phase2b - avg_phase1) / avg_phase1) * 100
                st.metric(
                    "Avg Ridership (Phase 2b)", 
                    f"{avg_phase2b:,.0f}",
                    f"{change:.1f}% vs Phase 1"
                )
        else:
            st.info("👆 Select corridors above to see comparison")
    
    # Tab 4: Data Explorer
    with tab4:
        st.header("🔍 Data Explorer")
        
        # Full data table with sorting
        st.subheader("Complete Corridor Data")
        
        # Column selector
        available_columns = comparison_df.columns.tolist()
        default_columns = [
            'corridor_id', 'phase2b_daily_riders', 'phase1_daily_riders',
            'phase2b_vs_phase1_pct', 'confidence', 'od_adjustment_factor',
            'mode_choice_effect'
        ]
        
        selected_columns = st.multiselect(
            "Select Columns to Display:",
            available_columns,
            default=[col for col in default_columns if col in available_columns]
        )
        
        if len(selected_columns) > 0:
            # Apply filters
            filtered_data = comparison_df[selected_columns].copy()
            
            # Sort options
            sort_by = st.selectbox("Sort by:", selected_columns)
            sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
            
            filtered_data = filtered_data.sort_values(
                by=sort_by,
                ascending=(sort_order == "Ascending")
            )
            
            # Display table
            st.dataframe(filtered_data, use_container_width=True, height=400)
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="apm_corridor_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON export
                json_str = filtered_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name="apm_corridor_analysis.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Select columns to display data")
    
    # Tab 5: Population & Trips
    with tab5:
        st.header("👥 Population & Trip Analysis")
        
        # Population metrics
        if synthetic_persons is not None and synthetic_households is not None:
            st.subheader("📊 Population Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_persons = len(synthetic_persons)
                st.metric("Total Population", f"{total_persons:,}")
            
            with col2:
                total_households = len(synthetic_households)
                st.metric("Total Households", f"{total_households:,}")
            
            with col3:
                avg_household_size = total_persons / total_households if total_households > 0 else 0
                st.metric("Avg Household Size", f"{avg_household_size:.2f}")
            
            with col4:
                if 'age' in synthetic_persons.columns:
                    avg_age = synthetic_persons['age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f}")
            
            st.markdown("---")
            
            # Population demographics
            st.subheader("📈 Demographics")
            
            tab_demo1, tab_demo2, tab_demo3 = st.tabs(["Age Distribution", "Household Income", "Geographic Distribution"])
            
            with tab_demo1:
                if 'age' in synthetic_persons.columns:
                    # Age distribution
                    age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
                    age_labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                    synthetic_persons['age_group'] = pd.cut(synthetic_persons['age'], bins=age_bins, labels=age_labels)
                    
                    age_dist = synthetic_persons['age_group'].value_counts().sort_index()
                    
                    fig = px.bar(
                        x=age_dist.index,
                        y=age_dist.values,
                        labels={'x': 'Age Group', 'y': 'Population'},
                        title='Population by Age Group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Age data not available")
            
            with tab_demo2:
                if 'income' in synthetic_households.columns:
                    # Income distribution
                    income_bins = [0, 25000, 50000, 75000, 100000, 150000, 999999]
                    income_labels = ['<$25k', '$25-50k', '$50-75k', '$75-100k', '$100-150k', '>$150k']
                    synthetic_households['income_group'] = pd.cut(synthetic_households['income'], bins=income_bins, labels=income_labels)
                    
                    income_dist = synthetic_households['income_group'].value_counts().sort_index()
                    
                    fig = px.bar(
                        x=income_dist.index,
                        y=income_dist.values,
                        labels={'x': 'Income Bracket', 'y': 'Households'},
                        title='Households by Income'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Income data not available")
            
            with tab_demo3:
                if parcels_enriched is not None:
                    # Try to aggregate population by zone
                    st.info("Geographic distribution can be viewed in the Interactive Map with Population Density overlay enabled")
                else:
                    st.info("Parcel-level geographic data not available")
        else:
            st.warning("⚠️ Population data not available. Ensure synthetic_persons.csv and synthetic_households.csv exist in data/processed/")
        
        st.markdown("---")
        
        # Trip analysis
        st.subheader("🚗 Trip Generation & Distribution")
        
        if od_matrix is not None:
            # OD matrix summary
            col1, col2, col3 = st.columns(3)
            
            # Get actual total trips from synthetic_trips file
            actual_trips_path = Path("data/processed/synthetic_trips_improved_v2.csv")
            if actual_trips_path.exists():
                trips_data = pd.read_csv(actual_trips_path)
                total_trips = len(trips_data)
                with col1:
                    st.metric("Total Daily Trips", f"{total_trips:,.0f}")
                
                # Also show OD flow info
                flow_cols = [c for c in od_matrix.columns if 'trip' in c.lower() or 'flow' in c.lower()]
                if flow_cols:
                    flow_col = flow_cols[0]
                    od_flow_total = od_matrix[flow_col].sum()
                    with col2:
                        st.metric("OD Flow Trips", f"{od_flow_total:,.0f}")
                        st.caption("(Major flows only)")
                        
                    with col3:
                        unique_dests = od_matrix['destination'].nunique() if 'destination' in od_matrix.columns else 0
                        st.metric("Trip Destinations", f"{unique_dests:,}")
            else:
                # Fall back to OD matrix if trips file not available
                flow_cols = [c for c in od_matrix.columns if 'trip' in c.lower() or 'flow' in c.lower()]
                if flow_cols:
                    flow_col = flow_cols[0]
                    total_trips = od_matrix[flow_col].sum()
                    with col1:
                        st.metric("Total Daily Trips (OD Flows)", f"{total_trips:,.0f}")
                    
                    with col2:
                        unique_origins = od_matrix['origin'].nunique() if 'origin' in od_matrix.columns else 0
                        st.metric("Trip Origins", f"{unique_origins:,}")
                    
                    with col3:
                        unique_dests = od_matrix['destination'].nunique() if 'destination' in od_matrix.columns else 0
                        st.metric("Trip Destinations", f"{unique_dests:,}")
            
            # Check if we have flow column for further analysis
            flow_cols = [c for c in od_matrix.columns if 'trip' in c.lower() or 'flow' in c.lower()]
            if flow_cols:
                flow_col = flow_cols[0]
                
                st.markdown("---")
                
                # Top OD pairs
                st.subheader("🔝 Top Origin-Destination Pairs")
                
                top_n_od = st.slider("Show Top N OD Pairs:", 5, 50, 20, step=5)
                
                top_od_pairs = od_matrix.nlargest(top_n_od, flow_col)
                
                # Format for display
                display_df = top_od_pairs.copy()
                if flow_col in display_df.columns:
                    display_df[flow_col] = display_df[flow_col].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Trip length distribution
                st.markdown("---")
                st.subheader("📏 Trip Distance Distribution")
                
                if 'distance' in od_matrix.columns or 'dist' in od_matrix.columns:
                    dist_col = 'distance' if 'distance' in od_matrix.columns else 'dist'
                    
                    # Create distance bins
                    distance_bins = [0, 1, 2, 3, 5, 10, 999]
                    distance_labels = ['<1 km', '1-2 km', '2-3 km', '3-5 km', '5-10 km', '>10 km']
                    
                    od_with_bins = od_matrix.copy()
                    od_with_bins['distance_group'] = pd.cut(od_with_bins[dist_col], bins=distance_bins, labels=distance_labels)
                    
                    # Aggregate trips by distance
                    dist_agg = od_with_bins.groupby('distance_group')[flow_col].sum()
                    
                    fig = px.bar(
                        x=dist_agg.index,
                        y=dist_agg.values,
                        labels={'x': 'Distance', 'y': 'Total Trips'},
                        title='Trip Distribution by Distance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Distance data not available in OD matrix")
            else:
                st.warning("⚠️ Trip flow column not found in OD matrix")
        else:
            st.warning("⚠️ OD matrix not available. Ensure od_matrix_calibrated_phase2b.csv exists in data/processed/")
        
        # Export section
        st.markdown("---")
        st.subheader("📥 Export Population & Trip Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if synthetic_persons is not None:
                csv = synthetic_persons.to_csv(index=False)
                st.download_button(
                    label="📄 Download Population Data",
                    data=csv,
                    file_name="population_summary.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if od_matrix is not None:
                csv = od_matrix.to_csv(index=False)
                st.download_button(
                    label="📄 Download OD Matrix",
                    data=csv,
                    file_name="od_matrix_summary.csv",
                    mime="text/csv"
                )
    
    # Tab 6: Model Parameters
    with tab6:
        st.header("⚙️ Model Parameters & Methodology")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Gravity Model Parameters")
            
            if gravity_params:
                st.json(gravity_params)
            else:
                st.info("Gravity parameters not available")
        
        with col2:
            st.subheader("🚇 Mode Choice Parameters")
            
            if mode_choice_params:
                st.json(mode_choice_params)
            else:
                st.info("Mode choice parameters not available")
        
        st.markdown("---")
        
        # Methodology explanation
        st.subheader("📖 Methodology Summary")
        
        methodology_tabs = st.tabs(["Phase 1", "Phase 2a", "Phase 2b"])
        
        with methodology_tabs[0]:
            st.markdown("""
            ### Phase 1: Baseline Model
            
            **Approach:**
            - Synthetic demand distribution (uniform across parcels)
            - Assumed distance decay (β = -1.0)
            - Generic NHTS mode choice coefficients
            - Fixed 1-mile stop spacing
            - Uniform 30% bus competition penalty
            
            **Confidence:** MEDIUM-HIGH (±15-20% uncertainty)
            
            **Limitations:**
            - No validation against local data
            - Optimistic assumptions
            - No geographic variation
            """)
        
        with methodology_tabs[1]:
            st.markdown("""
            ### Phase 2a: Improved Realism
            
            **Improvements:**
            - Gravity model for destination weighting (employment-based)
            - Time-of-day variation (peak/off-peak)
            - Zone-based bus competition (15-40% by density)
            
            **Confidence:** HIGH (±10-15% uncertainty)
            
            **Remaining Gaps:**
            - OD patterns still synthetic
            - Distance decay assumed
            - Mode choice generic
            """)
        
        with methodology_tabs[2]:
            st.markdown("""
            ### Phase 2b: Full Reality Grounding
            
            **Improvements:**
            - Empirical OD matrix (500+ respondent survey)
            - Doubly-constrained distance decay calibration (β = -1.25)
            - Hybrid mode choice (survey + literature triangulation)
            - All components validated against data
            
            **Confidence:** VERY HIGH (±5-10% uncertainty)
            
            **Key Changes:**
            - OD effect: -18% (empirical vs synthetic)
            - Distance decay: -9% (calibrated vs assumed)
            - Mode choice: -7% (local vs generic)
            - **Total: -34.6% reduction** (all data-backed)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><b>APM Corridor Analysis Interactive Visualizer</b></p>
        <p>Phase 2b Complete | Confidence: VERY HIGH | Generated: January 12, 2026</p>
        <p>📊 Data-driven | ✅ Validated | 🔍 Transparent</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
