#!/usr/bin/env python3
"""
Two-stage gabion optimization:
1. Find best LOCATIONS using a reference height
2. For each location, pick optimal HEIGHT to maximize metric

Usage:
  python optimize_locations_then_height.py --budget 100000
"""

import argparse
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import substring
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import functions from the main optimizer
from optimize_gabion_placement import (
    extract_contours_from_dem,
    get_contour_segments,
    evaluate_gabion_placement,
    get_upslope_buffer,
    calculate_slope,
    export_slope_raster
)


def find_best_locations(dem_path: str, wall_length: float = 100,
                        min_elev: float = 1070, max_elev: float = 1110,
                        min_spacing: float = 50, min_elev_diff: float = 10,
                        reference_height: float = 1.0,
                        max_locations: int = 20) -> list:
    """
    Find best gabion locations using a reference wall height.

    Returns list of segments (locations) sorted by metric.
    """
    print(f"Finding best locations (reference height: {reference_height}m)...")

    contours = extract_contours_from_dem(dem_path, interval=1.0,
                                          min_elev=min_elev, max_elev=max_elev)
    print(f"  Found {len(contours)} contour lines")

    segments = get_contour_segments(contours, wall_length)
    print(f"  Found {len(segments)} potential segments")

    # Evaluate all segments with reference height
    print(f"  Evaluating segments...")
    evaluated = []
    for i, segment in enumerate(segments):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(segments)}...")
        result = evaluate_gabion_placement(dem_path, segment, reference_height, 0.5)
        if result and result['fill_volume'] > 0:
            # Store the segment info for later re-evaluation at different heights
            result['segment'] = segment
            evaluated.append(result)

    print(f"  {len(evaluated)} segments with positive fill")

    # Sort by fill_efficiency
    evaluated.sort(key=lambda x: x.get('fill_efficiency', 0), reverse=True)

    # Greedy selection with spacing constraints
    selected = []
    for result in evaluated:
        result_elev = result['elevation']
        line = result['geometry']

        compatible = True
        for sel in selected:
            sel_elev = sel['elevation']
            elev_diff = abs(result_elev - sel_elev)

            if elev_diff == 0:
                # Same contour - check line distance
                if line.distance(sel['geometry']) < min_spacing:
                    compatible = False
                    break
            elif elev_diff < min_elev_diff:
                # Adjacent contours - too close
                compatible = False
                break

        if compatible:
            selected.append(result)
            if len(selected) >= max_locations:
                break

    print(f"  Selected {len(selected)} locations")
    return selected


def optimize_height_for_location(dem_path: str, segment: dict,
                                  heights: list = [0.5, 0.75, 1.0, 1.25, 1.5],
                                  metric: str = 'fill_efficiency') -> dict:
    """
    For a given location, find the optimal wall height.

    Returns the best result with optimal height.
    """
    best_result = None
    best_metric_value = -1

    for height in heights:
        result = evaluate_gabion_placement(dem_path, segment, height, 0.5)
        if result and result.get(metric, 0) > best_metric_value:
            best_metric_value = result.get(metric, 0)
            best_result = result
            best_result['optimal_height'] = height

    return best_result


def calc_cost(result):
    """Calculate total cost for a gabion."""
    fill_cost = result['fill_volume'] * 25
    rock_cost = result['rock_volume'] * 30
    mesh_area = result['wall_length'] * (2 * result['wall_height'] + 0.5)
    mesh_cost = mesh_area * 15
    return fill_cost + rock_cost + mesh_cost


def build_optimal_project(dem_path: str, locations: list, budget: float,
                          heights: list, metric: str) -> list:
    """
    Build project by selecting locations and optimal heights within budget.
    """
    print(f"\nOptimizing heights for each location...")

    # For each location, find optimal height
    optimized = []
    for i, loc in enumerate(locations):
        segment = loc['segment']
        best = optimize_height_for_location(dem_path, segment, heights, metric)
        if best:
            best['cost'] = calc_cost(best)
            best['location_rank'] = i + 1
            optimized.append(best)
            print(f"  Location {i+1} @ {loc['elevation']:.0f}m: "
                  f"optimal height = {best['wall_height']:.2f}m, "
                  f"flat area = {best['flat_area']:.0f}m², "
                  f"cost = {best['cost']:,.0f}лв")

    # Select within budget
    print(f"\nSelecting within {budget:,.0f}лв budget...")
    selected = []
    remaining = budget

    # Sort by metric value (descending)
    optimized.sort(key=lambda x: x.get(metric, 0), reverse=True)

    for result in optimized:
        if result['cost'] <= remaining:
            selected.append(result)
            remaining -= result['cost']
            print(f"  Added: {result['elevation']:.0f}m @ {result['wall_height']:.1f}m "
                  f"({result['flat_area']:.0f}m², {result['cost']:,.0f}лв)")

    return selected


def main():
    parser = argparse.ArgumentParser(description='Two-stage gabion optimization')
    parser.add_argument('--budget', '-b', type=float, default=100000,
                        help='Total budget in лв (default: 100000)')
    parser.add_argument('--metric', '-m', type=str, default='fill_efficiency',
                        choices=['fill_efficiency', 'flat_area', 'material_efficiency'],
                        help='Optimization metric (default: fill_efficiency)')
    parser.add_argument('--min-spacing', '-s', type=float, default=50,
                        help='Minimum line spacing in meters (default: 50)')
    parser.add_argument('--export', '-e', action='store_true',
                        help='Export results as GeoPackage')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    dem_path = script_dir.parent / "reprojected.tif"

    print("=" * 60)
    print("TWO-STAGE GABION OPTIMIZATION")
    print("=" * 60)
    print(f"Budget: {args.budget:,.0f} лв")
    print(f"Metric: {args.metric}")
    print()

    # Stage 1: Find best locations
    locations = find_best_locations(
        str(dem_path),
        min_spacing=args.min_spacing,
        max_locations=15
    )

    # Stage 2: Optimize height for each location within budget
    heights = [0.5, 0.75, 1.0, 1.25, 1.5]
    selected = build_optimal_project(
        str(dem_path),
        locations,
        args.budget,
        heights,
        args.metric
    )

    # Summary
    total_cost = sum(r['cost'] for r in selected)
    total_area = sum(r['flat_area'] for r in selected)

    print()
    print("=" * 60)
    print("FINAL PROJECT")
    print("=" * 60)
    print(f"Total gabions: {len(selected)}")
    print(f"Total cost: {total_cost:,.0f} лв")
    print(f"Reserve: {args.budget - total_cost:,.0f} лв")
    print(f"Total flat area: {total_area:,.0f} m²")
    if total_area > 0:
        print(f"Average cost: {total_cost/total_area:.0f} лв/m²")
    print()
    print("GABIONS:")
    print("-" * 60)
    print(f"{'#':<4} {'Elev':<8} {'Height':<8} {'Area m²':<10} {'Cost лв':<12}")
    print("-" * 60)

    for i, r in enumerate(selected, 1):
        print(f"{i:<4} {r['elevation']:<8.0f} {r['wall_height']:<8.1f} "
              f"{r['flat_area']:<10.0f} {r['cost']:<12,.0f}")

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_area:<10.0f} {total_cost:<12,.0f}")

    # Export
    if args.export and selected:
        # Remove fill_mask before export
        for r in selected:
            if 'fill_mask' in r:
                del r['fill_mask']
            if 'segment' in r:
                del r['segment']

        with rasterio.open(dem_path) as dem:
            crs = dem.crs

        gdf = gpd.GeoDataFrame(selected, crs=crs)
        output_path = script_dir / f"optimal_project_{args.metric}.gpkg"
        gdf.to_file(output_path, driver='GPKG')
        print(f"\nExported: {output_path}")


if __name__ == "__main__":
    main()
