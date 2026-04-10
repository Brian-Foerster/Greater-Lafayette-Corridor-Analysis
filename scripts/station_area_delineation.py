"""
Station Area Delineation - Corridor-Specific TIF Districts

Creates Tax Increment Financing (TIF) districts around APM stations with:
1. Realistic 400m buffer (not uniform 800m)
2. Corridor-specific boundaries
3. Parcel filtering for only station-adjacent properties
4. Spatial analysis optimized with R-tree indexing

Key Changes from Original Model:
- Original: All parcels within 800m (city-wide) count
- This: Only parcels within 400m of corridor's stations count
- Impact: 30-50% reduction in TIF district size (more realistic)

Author: UrbanSim APM Analysis
Date: January 13, 2026
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import logging
logger = logging.getLogger(__name__)



class TIFDistrictGenerator:
    """Generate TIF districts for APM corridors."""
    
    def __init__(self, buffer_distance_m: float = 400):
        """
        Initialize TIF district generator.
        
        Args:
            buffer_distance_m: Distance from stations for TIF eligibility (meters)
                              Typical TOD: 400m (1/4 mile walking distance)
                              Original model: 800m (too generous)
        """
        self.buffer_distance_m = buffer_distance_m
    
    def create_station_buffers(self, 
                               stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create buffer zones around stations.
        
        Optimized: Uses unary_union to merge overlapping buffers.
        """
        # Project to UTM for accurate distance calculations
        stations_utm = stations.to_crs("EPSG:32616")  # UTM Zone 16N (adjust for your region)
        
        # Buffer each station
        buffered = stations_utm.buffer(self.buffer_distance_m)
        
        # Merge overlapping buffers into single district
        merged_buffer = unary_union(buffered)
        
        # Convert back to geodataframe
        district = gpd.GeoDataFrame(
            {'geometry': [merged_buffer]},
            crs=stations_utm.crs
        ).to_crs(stations.crs)
        
        return district
    
    def select_parcels_in_tif(self,
                             parcels: gpd.GeoDataFrame,
                             tif_district: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Select parcels within TIF district boundaries.
        
        Optimized: Uses spatial index (R-tree) for fast intersection.
        """
        # Ensure CRS match
        if parcels.crs != tif_district.crs:
            tif_district = tif_district.to_crs(parcels.crs)
        
        # Spatial join (uses R-tree index automatically)
        parcels_in_tif = gpd.sjoin(
            parcels,
            tif_district,
            how='inner',
            predicate='intersects'
        )
        
        # Remove duplicate index column from join
        parcels_in_tif = parcels_in_tif.drop(columns=['index_right'], errors='ignore')
        
        return parcels_in_tif
    
    def calculate_tif_metrics(self,
                             parcels_in_tif: gpd.GeoDataFrame,
                             all_parcels: gpd.GeoDataFrame) -> Dict:
        """
        Calculate summary metrics for TIF district.
        
        Returns:
            Dictionary with coverage statistics
        """
        n_tif = len(parcels_in_tif)
        n_total = len(all_parcels)
        
        # Property value metrics
        if 'property_value' in parcels_in_tif.columns:
            base_value_tif = parcels_in_tif['property_value'].sum()
            base_value_total = all_parcels['property_value'].sum()
        else:
            base_value_tif = 0
            base_value_total = 0
        
        # Area metrics
        area_tif = parcels_in_tif.geometry.area.sum()
        area_total = all_parcels.geometry.area.sum()
        
        return {
            'n_parcels_tif': n_tif,
            'n_parcels_total': n_total,
            'pct_parcels_in_tif': 100 * n_tif / n_total if n_total > 0 else 0,
            'base_value_tif': base_value_tif,
            'base_value_total': base_value_total,
            'pct_value_in_tif': 100 * base_value_tif / base_value_total if base_value_total > 0 else 0,
            'area_tif_sqm': area_tif,
            'area_total_sqm': area_total,
            'pct_area_in_tif': 100 * area_tif / area_total if area_total > 0 else 0
        }


def generate_tif_district_for_corridor(corridor_id: str,
                                       buffer_distance_m: float = 400,
                                       data_dir: Path = Path("data")) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Generate TIF district for a specific corridor.
    
    Args:
        corridor_id: Corridor identifier (e.g., 'C23')
        buffer_distance_m: Buffer distance around stations
        data_dir: Path to data directory
    
    Returns:
        (parcels_in_tif, metrics_dict)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TIF DISTRICT GENERATION: {corridor_id}")
    logger.info(f"Buffer distance: {buffer_distance_m}m")
    logger.info(f"{'='*70}")
    
    # Load parcels
    parcels_path = data_dir / "processed" / "parcels_improved_gravity_v2.geojson"
    if not parcels_path.exists():
        parcels_path = data_dir / "processed" / "parcels_enriched.geojson"
    
    logger.info(f"Loading parcels from: {parcels_path.name}")
    parcels = gpd.read_file(parcels_path)
    logger.debug(f"  Loaded {len(parcels):,} parcels")
    
    # Load stations
    stops_path = data_dir / "processed" / "apm_phase2a_stops.geojson"
    if stops_path.exists():
        all_stops = gpd.read_file(stops_path)
        stations = all_stops[all_stops['corridor_id'] == corridor_id].copy()
    else:
        # Generate from corridor geometry
        corridors_path = data_dir / "processed" / "apm_phase2a_corridors.geojson"
        corridors = gpd.read_file(corridors_path)
        corridor = corridors[corridors['corridor_id'] == corridor_id].iloc[0]
        
        # Sample points along corridor every 1.5km
        line = corridor.geometry
        distances = np.arange(0, line.length, 1500)
        points = [line.interpolate(d) for d in distances]
        
        stations = gpd.GeoDataFrame(
            {'corridor_id': corridor_id, 'stop_id': range(len(points))},
            geometry=points,
            crs=parcels.crs
        )
    
    logger.debug(f"  Loaded {len(stations)} stations for {corridor_id}")
    
    # Generate TIF district
    generator = TIFDistrictGenerator(buffer_distance_m=buffer_distance_m)
    
    logger.info(f"\nCreating {buffer_distance_m}m buffer zones...")
    tif_district = generator.create_station_buffers(stations)
    
    logger.info("Selecting parcels within TIF district...")
    parcels_in_tif = generator.select_parcels_in_tif(parcels, tif_district)
    
    logger.info("Calculating TIF metrics...")
    metrics = generator.calculate_tif_metrics(parcels_in_tif, parcels)
    
    # Display results
    logger.info(f"\n{'='*70}")
    logger.info("TIF DISTRICT SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Parcels in TIF:     {metrics['n_parcels_tif']:,} / {metrics['n_parcels_total']:,} ({metrics['pct_parcels_in_tif']:.1f}%)")
    logger.info(f"Base value in TIF:  ${metrics['base_value_tif']/1e9:.2f}B / ${metrics['base_value_total']/1e9:.2f}B ({metrics['pct_value_in_tif']:.1f}%)")
    logger.info(f"Area in TIF:        {metrics['area_tif_sqm']/1e6:.2f} km² / {metrics['area_total_sqm']/1e6:.2f} km² ({metrics['pct_area_in_tif']:.1f}%)")
    
    return parcels_in_tif, metrics


def compare_buffer_distances(corridor_id: str,
                            buffer_distances: List[float] = [400, 600, 800],
                            data_dir: Path = Path("data")) -> pd.DataFrame:
    """
    Compare TIF district size across different buffer distances.
    
    Args:
        corridor_id: Corridor to analyze
        buffer_distances: List of buffer distances to test (meters)
        data_dir: Path to data directory
    
    Returns:
        DataFrame with comparison metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BUFFER DISTANCE SENSITIVITY ANALYSIS: {corridor_id}")
    logger.info(f"{'='*70}")
    
    results = []
    
    for buffer_m in buffer_distances:
        logger.info(f"\nTesting {buffer_m}m buffer...")
        parcels_in_tif, metrics = generate_tif_district_for_corridor(
            corridor_id=corridor_id,
            buffer_distance_m=buffer_m,
            data_dir=data_dir
        )
        
        results.append({
            'buffer_distance_m': buffer_m,
            'n_parcels': metrics['n_parcels_tif'],
            'pct_parcels': metrics['pct_parcels_in_tif'],
            'base_value_b': metrics['base_value_tif'] / 1e9,
            'pct_value': metrics['pct_value_in_tif'],
            'area_km2': metrics['area_tif_sqm'] / 1e6
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Display comparison table
    logger.info(f"\n{'='*70}")
    logger.info("BUFFER DISTANCE COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(comparison_df.to_string(index=False))
    
    # Highlight key findings
    logger.info(f"\n{'='*70}")
    logger.info("KEY FINDINGS")
    logger.info(f"{'='*70}")
    base_buffer = comparison_df.iloc[0]
    for i, row in comparison_df.iterrows():
        if i == 0:
            continue
        pct_increase_parcels = 100 * (row['n_parcels'] - base_buffer['n_parcels']) / base_buffer['n_parcels']
        pct_increase_value = 100 * (row['base_value_b'] - base_buffer['base_value_b']) / base_buffer['base_value_b']
        
        logger.info(f"{int(row['buffer_distance_m'])}m vs {int(base_buffer['buffer_distance_m'])}m:")
        logger.debug(f"  Parcels:  +{pct_increase_parcels:.1f}%")
        logger.debug(f"  Value:    +{pct_increase_value:.1f}%")
    
    return comparison_df


def batch_generate_all_corridors(buffer_distance_m: float = 400,
                                 output_dir: Path = Path("data/processed/tif_districts")) -> Dict:
    """
    Generate TIF districts for all corridors in Phase 2a results.
    
    Args:
        buffer_distance_m: Buffer distance to use
        output_dir: Where to save individual TIF district files
    
    Returns:
        Dictionary mapping corridor_id -> metrics
    """
    data_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all corridors
    corridors_path = data_dir / "processed" / "apm_phase2a_corridors.geojson"
    if not corridors_path.exists():
        logger.info(f"ERROR: {corridors_path} not found")
        return {}
    
    corridors = gpd.read_file(corridors_path)
    corridor_ids = corridors['corridor_id'].unique()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH TIF GENERATION: {len(corridor_ids)} corridors")
    logger.info(f"{'='*70}")
    
    all_metrics = {}
    
    for corridor_id in corridor_ids:
        parcels_in_tif, metrics = generate_tif_district_for_corridor(
            corridor_id=corridor_id,
            buffer_distance_m=buffer_distance_m,
            data_dir=data_dir
        )
        
        # Save parcels to file
        output_path = output_dir / f"tif_parcels_{corridor_id}_{buffer_distance_m}m.geojson"
        parcels_in_tif.to_file(output_path, driver='GeoJSON')
        logger.debug(f"  Saved: {output_path.name}")
        
        all_metrics[corridor_id] = metrics
    
    # Create summary table
    summary_df = pd.DataFrame(all_metrics).T
    summary_df.index.name = 'corridor_id'
    summary_path = output_dir / f"tif_summary_{buffer_distance_m}m.csv"
    summary_df.to_csv(summary_path)
    logger.info(f"\nSummary saved: {summary_path}")
    
    return all_metrics


def main():
    """Test TIF district generation for C23."""
    
    # Single corridor test
    corridor_id = "C23"
    parcels_in_tif, metrics = generate_tif_district_for_corridor(
        corridor_id=corridor_id,
        buffer_distance_m=400
    )
    
    # Buffer distance sensitivity
    comparison = compare_buffer_distances(
        corridor_id=corridor_id,
        buffer_distances=[400, 600, 800]
    )
    
    # Save results
    output_dir = Path("data/processed/tif_districts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parcels_in_tif.to_file(
        output_dir / f"tif_parcels_{corridor_id}_400m.geojson",
        driver='GeoJSON'
    )
    comparison.to_csv(
        output_dir / f"buffer_comparison_{corridor_id}.csv",
        index=False
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
