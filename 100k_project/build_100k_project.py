#!/usr/bin/env python3
"""
Build the 100k лв gabion project layer.

Combines optimal 0.5m and 1.0m gabion placements into a single
GeoPackage for visualization and planning.

Budget allocation:
  - Phase 1: All 7 × 0.5m walls (best cost efficiency)
  - Phase 2: Top 3 × 1.0m walls (remaining budget)

Usage:
  python build_100k_project.py
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path


def build_project(walls_05_path: str, walls_10_path: str,
                  output_path: str, budget: float = 100000) -> gpd.GeoDataFrame:
    """
    Build combined project layer from separate height optimizations.

    Args:
        walls_05_path: Path to 0.5m walls GeoPackage
        walls_10_path: Path to 1.0m walls GeoPackage
        output_path: Output GeoPackage path
        budget: Total budget in лв

    Returns:
        Combined GeoDataFrame
    """
    # Load both layers
    walls_05 = gpd.read_file(walls_05_path)
    walls_10 = gpd.read_file(walls_10_path)

    # Add phase/height labels
    walls_05['phase'] = 1
    walls_05['wall_height_m'] = 0.5
    walls_10['phase'] = 2
    walls_10['wall_height_m'] = 1.0

    # Calculate costs
    def calc_cost(row):
        fill_cost = row['fill_volume'] * 25
        rock_cost = row['rock_volume'] * 30
        mesh_area = row['wall_length'] * (2 * row['wall_height_m'] + 0.5)
        mesh_cost = mesh_area * 15
        return fill_cost + rock_cost + mesh_cost

    walls_05['cost_lv'] = walls_05.apply(calc_cost, axis=1)
    walls_10['cost_lv'] = walls_10.apply(calc_cost, axis=1)

    # Greedy selection within budget with cross-height elevation filtering
    selected = []
    remaining_budget = budget
    min_elev_diff = 10  # Minimum elevation difference between any gabions
    min_line_spacing = 50  # Minimum line-to-line distance

    def is_compatible(candidate, selected_list):
        """Check if candidate gabion is compatible with already selected ones."""
        candidate_elev = candidate['elevation']
        candidate_geom = candidate.geometry

        for sel in selected_list:
            sel_elev = sel['elevation']
            elev_diff = abs(candidate_elev - sel_elev)

            if elev_diff == 0:
                # Same contour - check line distance
                if candidate_geom.distance(sel.geometry) < min_line_spacing:
                    return False
            elif elev_diff < min_elev_diff:
                # Adjacent contours - too close geographically
                return False
        return True

    # Phase 1: Add 0.5m walls
    for idx, row in walls_05.iterrows():
        if row['cost_lv'] <= remaining_budget and is_compatible(row, selected):
            selected.append(row)
            remaining_budget -= row['cost_lv']

    # Phase 2: Add 1.0m walls with remaining budget (also checking against Phase 1)
    for idx, row in walls_10.iterrows():
        if row['cost_lv'] <= remaining_budget and is_compatible(row, selected):
            row_copy = row.copy()
            selected.append(row_copy)
            remaining_budget -= row['cost_lv']

    # Build combined dataframe
    project = gpd.GeoDataFrame(selected, crs=walls_05.crs)
    project['project_rank'] = range(1, len(project) + 1)

    # Save
    project.to_file(output_path, driver="GPKG")

    return project


def print_summary(project: gpd.GeoDataFrame, budget: float = 100000):
    """Print project summary."""
    total_cost = project['cost_lv'].sum()
    total_area = project['flat_area'].sum()
    phase1 = project[project['phase'] == 1]
    phase2 = project[project['phase'] == 2]

    print("=" * 60)
    print("100K GABION PROJECT SUMMARY")
    print("=" * 60)
    print()
    print(f"Budget: {budget:,.0f} лв")
    print(f"Spent:  {total_cost:,.0f} лв")
    print(f"Reserve: {budget - total_cost:,.0f} лв")
    print()
    print(f"Total gabions: {len(project)}")
    print(f"  Phase 1 (0.5m walls): {len(phase1)}")
    print(f"  Phase 2 (1.0m walls): {len(phase2)}")
    print()
    print(f"Total flat area created: {total_area:,.0f} m²")
    print(f"Average cost: {total_cost/total_area:.0f} лв/m²")
    print()
    print("INDIVIDUAL GABIONS:")
    print("-" * 60)
    print(f"{'#':<4} {'Phase':<6} {'Height':<8} {'Elev':<8} {'Area m²':<10} {'Cost лв':<12}")
    print("-" * 60)

    for _, row in project.iterrows():
        print(f"{row['project_rank']:<4} {row['phase']:<6} {row['wall_height_m']:<8} "
              f"{row['elevation']:<8.0f} {row['flat_area']:<10.0f} {row['cost_lv']:<12,.0f}")

    print("-" * 60)
    print(f"{'TOTAL':<28} {total_area:<10.0f} {total_cost:<12,.0f}")
    print()


def main():
    script_dir = Path(__file__).parent  # 100k_project/

    # Input files (from optimizer runs in same directory)
    walls_05_path = script_dir / "optimal_gabions_0.5m_fill_efficiency.gpkg"
    walls_10_path = script_dir / "optimal_gabions_1.0m_fill_efficiency.gpkg"

    # Output
    output_path = script_dir / "project_100k_gabions.gpkg"

    # Check inputs exist
    if not walls_05_path.exists():
        print(f"Missing: {walls_05_path}")
        print("Run: python optimize_gabion_placement.py --height 0.5 --top 10 --min-spacing 50 --export")
        return

    if not walls_10_path.exists():
        print(f"Missing: {walls_10_path}")
        print("Run: python optimize_gabion_placement.py --height 1.0 --top 10 --min-spacing 50 --export")
        return

    # Build project
    project = build_project(
        str(walls_05_path),
        str(walls_10_path),
        str(output_path),
        budget=100000
    )

    print_summary(project)
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
