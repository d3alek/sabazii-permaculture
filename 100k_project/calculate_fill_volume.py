#!/usr/bin/env python3
"""
Calculate fill volumes for gabion terraces.

Inputs:
  - DEM raster (reprojected.tif)
  - Terraces polygons (terraces_1m.gpkg)
  - Gabions lines with base_elev (gabions.gpkg)

Outputs:
  - Fill volume per terrace (m³)
  - Gabion rock volume (m³)
  - Summary table

Usage:
  python calculate_fill_volume.py --height 0.5
  python calculate_fill_volume.py --height 1.0
  python calculate_fill_volume.py --height 1.5
  python calculate_fill_volume.py --height 0.5 1.0 1.5  # compare all
"""

import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path


def calculate_fill_volumes(dem_path: str, terraces_path: str, gabions_path: str,
                           wall_height: float, wall_width: float = 0.5,
                           verbose: bool = True) -> dict:
    """
    Calculate fill volume for each terrace.

    Fill occurs where: target_elev > dem_elev
    Fill depth = target_elev - dem_elev (where positive)
    Volume = sum(fill_depth) * cell_area

    Target elevation is calculated as: gabion base_elev + wall_height
    """
    # Load DEM
    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs
        pixel_width = dem.res[0]
        pixel_height = dem.res[1]
        cell_area = pixel_width * pixel_height
        dem_nodata = dem.nodata

        if verbose:
            print(f"DEM: {dem_path}")
            print(f"  Resolution: {pixel_width} x {pixel_height} m")
            print(f"  Cell area: {cell_area} m²")
            print()

    # Load gabions to get base elevations
    gabions = gpd.read_file(gabions_path, layer='gabion_walls')
    gabion_elevs = {g['name']: g['base_elev'] for _, g in gabions.iterrows()}

    # Map terrace to its retaining gabion (terrace_1 -> gabion_1080, etc.)
    # We use the naming convention: terrace_N is retained by the Nth lowest gabion
    sorted_gabions = sorted(gabion_elevs.items(), key=lambda x: x[1])
    terrace_to_gabion = {f"terrace_{i+1}": name for i, (name, _) in enumerate(sorted_gabions)}

    # Load terraces
    terraces = gpd.read_file(terraces_path)

    # Ensure same CRS
    if terraces.crs != dem_crs:
        terraces = terraces.to_crs(dem_crs)

    if verbose:
        print(f"Terraces: {terraces_path}")
        print(f"  Count: {len(terraces)}")
        print(f"  Wall height: {wall_height} m")
        print()

    results = []

    with rasterio.open(dem_path) as dem:
        for idx, terrace in terraces.iterrows():
            name = terrace.get('name', f'terrace_{idx}')

            # Calculate target elevation from gabion base + wall height
            gabion_name = terrace_to_gabion.get(name)
            if gabion_name and gabion_name in gabion_elevs:
                target_elev = gabion_elevs[gabion_name] + wall_height
            else:
                # Fallback: try to get from terrace attribute
                target_elev = terrace.get('target_elev')

            if target_elev is None or np.isnan(target_elev):
                if verbose:
                    print(f"  {name}: skipped (no target elevation)")
                continue

            # Mask DEM to terrace polygon
            try:
                masked_dem, masked_transform = mask(
                    dem,
                    [terrace.geometry],
                    crop=True,
                    nodata=dem_nodata
                )
            except ValueError as e:
                if verbose:
                    print(f"  {name}: skipped ({e})")
                continue

            # Get the DEM values (first band)
            dem_values = masked_dem[0]

            # Create valid mask (not nodata)
            if dem_nodata is not None:
                valid_mask = dem_values != dem_nodata
            else:
                valid_mask = ~np.isnan(dem_values)

            # Calculate fill depth where target > dem
            fill_depth = np.where(
                valid_mask & (target_elev > dem_values),
                target_elev - dem_values,
                0
            )

            # Sum and convert to volume
            total_fill_depth = np.sum(fill_depth)
            volume_m3 = total_fill_depth * cell_area

            # Stats
            valid_cells = np.sum(valid_mask)
            fill_cells = np.sum(fill_depth > 0)
            max_fill = np.max(fill_depth) if fill_cells > 0 else 0
            avg_fill = np.mean(fill_depth[fill_depth > 0]) if fill_cells > 0 else 0

            results.append({
                'name': name,
                'target_elev': target_elev,
                'valid_cells': valid_cells,
                'fill_cells': fill_cells,
                'max_fill_m': max_fill,
                'avg_fill_m': avg_fill,
                'volume_m3': volume_m3
            })

            if verbose:
                print(f"  {name}:")
                print(f"    Target elevation: {target_elev:.1f} m")
                print(f"    Cells with fill: {fill_cells} / {valid_cells}")
                print(f"    Max fill depth: {max_fill:.2f} m")
                print(f"    Avg fill depth: {avg_fill:.2f} m")
                print(f"    Fill volume: {volume_m3:.1f} m³")
                print()

    return results


def calculate_gabion_volumes(gabions_path: str, wall_height: float,
                             wall_width: float = 0.5, verbose: bool = True) -> dict:
    """
    Calculate gabion rock volume.

    Rock volume = length × height × width
    """
    gabions = gpd.read_file(gabions_path, layer='gabion_walls')

    if verbose:
        print(f"Gabions: {gabions_path}")
        print(f"  Count: {len(gabions)}")
        print(f"  Wall dimensions: {wall_height} × {wall_width} m")
        print()

    results = []
    total_length = 0
    total_rock = 0

    for idx, gabion in gabions.iterrows():
        name = gabion.get('name', f'gabion_{idx}')
        length = gabion.geometry.length
        base_elev = gabion.get('base_elev', 0)

        rock_volume = length * wall_height * wall_width

        results.append({
            'name': name,
            'base_elev': base_elev,
            'length_m': length,
            'height_m': wall_height,
            'width_m': wall_width,
            'rock_volume_m3': rock_volume
        })

        total_length += length
        total_rock += rock_volume

        if verbose:
            print(f"  {name}:")
            print(f"    Base elevation: {base_elev:.0f} m")
            print(f"    Length: {length:.1f} m")
            print(f"    Rock volume: {rock_volume:.1f} m³")
            print()

    if verbose:
        print(f"  TOTAL:")
        print(f"    Total length: {total_length:.1f} m")
        print(f"    Total rock: {total_rock:.1f} m³")
        print()

    return results


def print_summary(fill_results: list, gabion_results: list, wall_height: float):
    """Print summary table."""
    print("=" * 60)
    print(f"SUMMARY (wall height: {wall_height:.1f} m)")
    print("=" * 60)
    print()

    print("FILL VOLUMES:")
    print("-" * 40)
    total_fill = 0
    for r in fill_results:
        print(f"  {r['name']:20s} {r['volume_m3']:>10.1f} m³")
        total_fill += r['volume_m3']
    print("-" * 40)
    print(f"  {'TOTAL':20s} {total_fill:>10.1f} m³")
    print()

    print("GABION ROCK:")
    print("-" * 40)
    total_rock = 0
    total_length = 0
    for r in gabion_results:
        print(f"  {r['name']:20s} {r['rock_volume_m3']:>10.1f} m³  ({r['length_m']:.0f} m)")
        total_rock += r['rock_volume_m3']
        total_length += r['length_m']
    print("-" * 40)
    print(f"  {'TOTAL':20s} {total_rock:>10.1f} m³  ({total_length:.0f} m)")
    print()

    print("COST ESTIMATE (rough):")
    print("-" * 40)
    fill_cost = total_fill * 25  # 25 лв/m³
    rock_cost = total_rock * 30  # 30 лв/m³
    mesh_area = total_length * (2 * wall_height + 2 * 0.5 + 2 * wall_height * 0.5 / total_length * len(gabion_results))  # rough
    mesh_cost = mesh_area * 15  # 15 лв/m²

    print(f"  Fill material:     {total_fill:>8.0f} m³ × 25 лв = {fill_cost:>10,.0f} лв")
    print(f"  Gabion rock:       {total_rock:>8.0f} m³ × 30 лв = {rock_cost:>10,.0f} лв")
    print(f"  Gabion mesh:       {mesh_area:>8.0f} m² × 15 лв = {mesh_cost:>10,.0f} лв")
    print("-" * 40)
    print(f"  {'TOTAL':20s}              {fill_cost + rock_cost + mesh_cost:>10,.0f} лв")
    print()


def export_modified_dem(dem_path: str, terraces_path: str, gabions_path: str,
                        wall_height: float, output_path: str) -> str:
    """
    Export a modified DEM showing terrain after gabion fill.

    Where fill occurs: elevation = target_elev (gabion top)
    Where no fill: elevation = original DEM
    """
    from rasterio.features import geometry_mask

    # Load gabions to get base elevations
    gabions = gpd.read_file(gabions_path, layer='gabion_walls')
    gabion_elevs = {g['name']: g['base_elev'] for _, g in gabions.iterrows()}
    sorted_gabions = sorted(gabion_elevs.items(), key=lambda x: x[1])
    terrace_to_gabion = {f"terrace_{i+1}": name for i, (name, _) in enumerate(sorted_gabions)}

    # Load terraces
    terraces = gpd.read_file(terraces_path)

    with rasterio.open(dem_path) as dem:
        # Read the DEM
        dem_data = dem.read(1)
        dem_nodata = dem.nodata
        profile = dem.profile.copy()

        # Ensure same CRS
        if terraces.crs != dem.crs:
            terraces = terraces.to_crs(dem.crs)

        # Create modified DEM (copy of original)
        modified = dem_data.copy().astype(np.float32)

        for idx, terrace in terraces.iterrows():
            name = terrace.get('name', f'terrace_{idx}')

            # Get target elevation
            gabion_name = terrace_to_gabion.get(name)
            if gabion_name and gabion_name in gabion_elevs:
                target_elev = gabion_elevs[gabion_name] + wall_height
            else:
                continue

            # Create mask for this terrace
            terrace_mask = geometry_mask(
                [terrace.geometry],
                out_shape=dem_data.shape,
                transform=dem.transform,
                invert=True  # True inside polygon
            )

            # Where inside terrace AND original < target: set to target
            fill_mask = terrace_mask & (dem_data < target_elev)
            if dem_nodata is not None:
                fill_mask = fill_mask & (dem_data != dem_nodata)

            modified[fill_mask] = target_elev

    # Write output
    profile.update(dtype=np.float32)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(modified, 1)

    return output_path


def run_calculation(dem_path: Path, terraces_path: Path, gabions_path: Path,
                    wall_height: float, wall_width: float = 0.5,
                    verbose: bool = True, export_dem: bool = False,
                    output_dir: Path = None) -> dict:
    """Run calculation for a single wall height."""
    fill_results = calculate_fill_volumes(
        dem_path, terraces_path, gabions_path,
        wall_height, wall_width, verbose
    )
    gabion_results = calculate_gabion_volumes(
        gabions_path, wall_height, wall_width, verbose
    )

    result = {
        'wall_height': wall_height,
        'fill_results': fill_results,
        'gabion_results': gabion_results,
        'modified_dem_path': None
    }

    if export_dem and output_dir:
        output_path = output_dir / f"dem_with_gabions_{wall_height:.1f}m.tif"
        export_modified_dem(
            dem_path, terraces_path, gabions_path,
            wall_height, str(output_path)
        )
        result['modified_dem_path'] = output_path
        if verbose:
            print(f"  Exported: {output_path}")

    return result


def print_comparison(all_results: list, wall_width: float):
    """Print comparison table for multiple wall heights."""
    print()
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    # Header
    heights = [r['wall_height'] for r in all_results]
    header = f"{'Item':<25}"
    for h in heights:
        header += f" {h:.1f}m wall".rjust(15)
    print(header)
    print("-" * (25 + 15 * len(heights)))

    # Fill volumes per terrace
    terrace_names = [f['name'] for f in all_results[0]['fill_results']]
    for tname in terrace_names:
        row = f"  Fill {tname:<19}"
        for r in all_results:
            vol = next((f['volume_m3'] for f in r['fill_results'] if f['name'] == tname), 0)
            row += f"{vol:>12.0f} m³"
        print(row)

    # Total fill
    row = f"  {'Fill TOTAL':<23}"
    for r in all_results:
        total = sum(f['volume_m3'] for f in r['fill_results'])
        row += f"{total:>12.0f} m³"
    print(row)
    print()

    # Gabion rock
    row = f"  {'Gabion rock':<23}"
    for r in all_results:
        total = sum(g['rock_volume_m3'] for g in r['gabion_results'])
        row += f"{total:>12.0f} m³"
    print(row)

    # Total length (same for all)
    total_length = sum(g['length_m'] for g in all_results[0]['gabion_results'])
    print(f"  {'Gabion length':<23}" + f"{total_length:>12.0f} m" * len(heights))
    print()

    # Cost estimates
    print("-" * (25 + 15 * len(heights)))
    row = f"  {'Fill cost (25лв/m³)':<23}"
    for r in all_results:
        total = sum(f['volume_m3'] for f in r['fill_results']) * 25
        row += f"{total:>11,.0f} лв"
    print(row)

    row = f"  {'Rock cost (30лв/m³)':<23}"
    for r in all_results:
        total = sum(g['rock_volume_m3'] for g in r['gabion_results']) * 30
        row += f"{total:>11,.0f} лв"
    print(row)

    row = f"  {'Mesh cost (15лв/m²)':<23}"
    for r in all_results:
        h = r['wall_height']
        # Mesh area: 2 sides + top + 2 ends per gabion
        mesh_per_m = 2 * h + wall_width  # simplified: 2 sides + top per linear meter
        total = total_length * mesh_per_m * 15
        row += f"{total:>11,.0f} лв"
    print(row)

    print("-" * (25 + 15 * len(heights)))
    row = f"  {'TOTAL COST':<23}"
    for r in all_results:
        h = r['wall_height']
        fill_cost = sum(f['volume_m3'] for f in r['fill_results']) * 25
        rock_cost = sum(g['rock_volume_m3'] for g in r['gabion_results']) * 30
        mesh_per_m = 2 * h + wall_width
        mesh_cost = total_length * mesh_per_m * 15
        total = fill_cost + rock_cost + mesh_cost
        row += f"{total:>11,.0f} лв"
    print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate fill volumes for gabion terraces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --height 0.5
  %(prog)s --height 1.0
  %(prog)s --height 0.5 1.0 1.5    # compare multiple heights
  %(prog)s --height 0.5 1.0 1.5 --width 0.5
        """
    )
    parser.add_argument(
        '--height', '-H',
        type=float,
        nargs='+',
        default=[1.0],
        help='Wall height(s) in meters (default: 1.0)'
    )
    parser.add_argument(
        '--width', '-W',
        type=float,
        default=0.5,
        help='Wall width in meters (default: 0.5)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show summary/comparison table'
    )
    parser.add_argument(
        '--export', '-e',
        action='store_true',
        help='Export modified DEM rasters (dem_with_gabions_Xm.tif)'
    )

    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent
    dem_path = project_dir / "reprojected.tif"
    terraces_path = project_dir.parent / "terraces_1m.gpkg"
    gabions_path = project_dir.parent / "gabions.gpkg"

    print("=" * 60)
    print("GABION TERRACE VOLUME CALCULATOR")
    print("=" * 60)

    all_results = []

    for height in args.height:
        print()
        print(f">>> WALL HEIGHT: {height} m <<<")
        print("-" * 60)

        result = run_calculation(
            dem_path, terraces_path, gabions_path,
            wall_height=height,
            wall_width=args.width,
            verbose=not args.quiet,
            export_dem=args.export,
            output_dir=project_dir
        )
        all_results.append(result)

        if not args.quiet and len(args.height) == 1:
            print_summary(
                result['fill_results'],
                result['gabion_results'],
                height
            )

    # Print comparison if multiple heights
    if len(args.height) > 1:
        print_comparison(all_results, args.width)

    # Print exported files
    if args.export:
        print()
        print("EXPORTED MODIFIED DEMs:")
        print("-" * 60)
        for r in all_results:
            if r['modified_dem_path']:
                print(f"  {r['modified_dem_path']}")
        print()
        print("To visualize in QGIS:")
        print("  1. Layer → Add Layer → Add Raster Layer")
        print("  2. Load the dem_with_gabions_*.tif files")
        print("  3. Raster → Analysis → Slope (for each)")
        print("  4. Compare slope maps side by side")
        print()


if __name__ == "__main__":
    main()
