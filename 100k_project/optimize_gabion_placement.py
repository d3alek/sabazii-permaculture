#!/usr/bin/env python3
"""
Optimize gabion wall placement.

Finds the best contour line segment for gabion placement by maximizing
a metric that balances fill efficiency and usable area created.

Usage:
  python optimize_gabion_placement.py --length 100 --height 1.0
  python optimize_gabion_placement.py --length 100 --height 0.5 1.0 1.5
"""

import argparse
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_contours_from_dem(dem_path: str, interval: float = 1.0,
                               min_elev: float = None, max_elev: float = None) -> gpd.GeoDataFrame:
    """Extract contour lines from DEM using matplotlib."""
    import matplotlib.pyplot as plt

    with rasterio.open(dem_path) as dem:
        data = dem.read(1)
        transform = dem.transform
        nodata = dem.nodata

        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)

        # Get coordinate arrays
        rows, cols = data.shape
        xs = transform[2] + transform[0] * np.arange(cols)
        ys = transform[5] + transform[4] * np.arange(rows)

        # Determine contour levels
        if min_elev is None:
            min_elev = float(np.floor(data.min()))
        if max_elev is None:
            max_elev = float(np.ceil(data.max()))

        levels = np.arange(min_elev, max_elev + interval, interval)

        # Extract contours
        fig, ax = plt.subplots()
        cs = ax.contour(xs, ys, data, levels=levels)
        plt.close(fig)

        # Convert to GeoDataFrame using allsegs (works with all matplotlib versions)
        contours = []
        for level_idx, level in enumerate(cs.levels):
            if level_idx < len(cs.allsegs):
                for seg in cs.allsegs[level_idx]:
                    if len(seg) > 1:
                        line = LineString(seg)
                        if line.is_valid and line.length > 10:  # Skip tiny fragments
                            contours.append({
                                'elevation': level,
                                'geometry': line
                            })

        gdf = gpd.GeoDataFrame(contours, crs=dem.crs)
        return gdf


def get_contour_segments(contours: gpd.GeoDataFrame, segment_length: float,
                         boundary: gpd.GeoDataFrame = None) -> list:
    """
    Extract all possible segments of given length from contour lines.
    Optionally clip to boundary.
    """
    segments = []

    for idx, row in contours.iterrows():
        line = row.geometry
        elevation = row['elevation']

        # Clip to boundary if provided
        if boundary is not None:
            line = line.intersection(boundary.unary_union)
            if line.is_empty:
                continue
            # Handle MultiLineString from intersection
            if line.geom_type == 'MultiLineString':
                lines = list(line.geoms)
            else:
                lines = [line]
        else:
            lines = [line]

        # Extract segments from each line
        for ln in lines:
            if ln.length < segment_length:
                continue

            # Sample segments along the line
            step = segment_length / 4  # Overlap for better coverage
            distance = 0
            while distance + segment_length <= ln.length:
                segment = substring(ln, distance, distance + segment_length)
                if segment.length >= segment_length * 0.95:  # Allow 5% tolerance
                    segments.append({
                        'elevation': elevation,
                        'start_distance': distance,
                        'geometry': segment
                    })
                distance += step

    return segments


def get_upslope_direction(line: LineString, dem_data: np.ndarray, transform) -> int:
    """
    Determine which side of the line is upslope.

    Returns:
        1 if left side is upslope, -1 if right side is upslope
    """
    from shapely.geometry import Polygon

    buffer_dist = 5.0  # Sample distance

    try:
        left_poly = line.buffer(buffer_dist, single_sided=True)
        right_poly = line.buffer(-buffer_dist, single_sided=True)
    except Exception:
        return 1  # Default to left

    def get_avg_elevation(poly):
        if poly is None or poly.is_empty or not poly.is_valid:
            return -9999
        try:
            mask = geometry_mask([poly], out_shape=dem_data.shape,
                                transform=transform, invert=True)
            if not np.any(mask):
                return -9999
            valid = dem_data[mask]
            valid = valid[valid > -9999]
            if len(valid) == 0:
                return -9999
            return np.mean(valid)
        except Exception:
            return -9999

    left_elev = get_avg_elevation(left_poly)
    right_elev = get_avg_elevation(right_poly)

    return 1 if left_elev > right_elev else -1


def get_contour_based_fill_zone(line: LineString, dem_data: np.ndarray, transform,
                                  base_elev: float, wall_height: float,
                                  max_upslope_dist: float = None) -> 'Polygon':
    """
    Create fill zone bounded by actual contour at target elevation.

    The fill zone is a polygon bounded by:
    - The gabion line (downslope edge)
    - The contour at base_elev + wall_height (upslope edge)
    - Lines connecting gabion endpoints to closest points on target contour (lateral edges)

    Args:
        line: Gabion line geometry
        dem_data: DEM array
        transform: Rasterio transform
        base_elev: Elevation of gabion line
        wall_height: Height of gabion wall
        max_upslope_dist: Maximum distance to search upslope (default: wall_height * 10)

    Returns:
        Polygon representing the fill zone
    """
    from shapely.geometry import Polygon, LineString as ShapelyLine
    from shapely.ops import substring
    import matplotlib.pyplot as plt

    if max_upslope_dist is None:
        max_upslope_dist = wall_height * 10

    target_elev = base_elev + wall_height

    # Get gabion line endpoints
    coords = list(line.coords)
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])

    # Extract contour at target elevation from DEM
    rows, cols = dem_data.shape
    xs = transform[2] + transform[0] * np.arange(cols)
    ys = transform[5] + transform[4] * np.arange(rows)

    fig, ax = plt.subplots()
    cs = ax.contour(xs, ys, dem_data, levels=[target_elev])
    plt.close(fig)

    # Collect contour segments at target elevation
    target_contour_segments = []
    if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
        for seg in cs.allsegs[0]:
            if len(seg) > 1:
                target_contour_segments.append(ShapelyLine(seg))

    if not target_contour_segments:
        raise ValueError(f"No contour found at target elevation {target_elev}m")

    # Find the contour segment closest to the gabion line
    closest_contour = min(target_contour_segments, key=lambda g: g.distance(line))

    # Check if the target contour is within reasonable distance
    if closest_contour.distance(line) > max_upslope_dist:
        raise ValueError(f"Target contour at {target_elev}m is too far ({closest_contour.distance(line):.1f}m) from gabion at {base_elev}m")

    # Find closest points on target contour to each gabion endpoint
    start_dist_on_contour = closest_contour.project(start_pt)
    end_dist_on_contour = closest_contour.project(end_pt)

    start_closest = closest_contour.interpolate(start_dist_on_contour)
    end_closest = closest_contour.interpolate(end_dist_on_contour)

    # Extract the portion of contour between these two closest points
    if start_dist_on_contour > end_dist_on_contour:
        start_dist_on_contour, end_dist_on_contour = end_dist_on_contour, start_dist_on_contour
        start_closest, end_closest = end_closest, start_closest

    contour_segment = substring(closest_contour, start_dist_on_contour, end_dist_on_contour)

    if contour_segment.is_empty or contour_segment.length == 0:
        raise ValueError(f"Empty contour segment for gabion at {base_elev}m")

    # Build the fill polygon: gabion line -> contour segment -> close
    contour_coords = list(contour_segment.coords)

    # Determine correct ordering: gabion end should connect to nearest contour end
    contour_start = Point(contour_coords[0])
    contour_end = Point(contour_coords[-1])

    # Build polygon: gabion start -> gabion end -> contour (in correct direction) -> back
    if contour_start.distance(end_pt) < contour_end.distance(end_pt):
        fill_coords = list(coords) + contour_coords + [coords[0]]
    else:
        fill_coords = list(coords) + list(reversed(contour_coords)) + [coords[0]]

    fill_poly = Polygon(fill_coords)
    if not fill_poly.is_valid:
        fill_poly = fill_poly.buffer(0)
    if not fill_poly.is_valid or fill_poly.is_empty or fill_poly.area == 0:
        raise ValueError(f"Could not create valid fill polygon for gabion at {base_elev}m")

    return fill_poly


def evaluate_gabion_placement(dem_path: str, segment: dict, wall_height: float,
                               wall_width: float = 0.5,
                               excluded_mask: np.ndarray = None) -> dict:
    """
    Evaluate a potential gabion placement.

    Args:
        dem_path: Path to DEM raster
        segment: Dict with 'geometry' and 'elevation'
        wall_height: Gabion wall height in meters
        wall_width: Gabion wall width in meters
        excluded_mask: Boolean array of already-claimed pixels (True = excluded)

    Returns metrics about fill volume, flat area created, etc.
    """
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        dem_nodata = dem.nodata
        transform = dem.transform
        pixel_area = abs(transform[0] * transform[4])

        line = segment['geometry']
        base_elev = segment['elevation']
        target_elev = base_elev + wall_height

        # Create contour-based fill zone (bounded by uphill contour and perpendicular edges)
        try:
            fill_zone = get_contour_based_fill_zone(
                line, dem_data, transform, base_elev, wall_height
            )
        except ValueError:
            # No valid fill zone could be created (e.g., no uphill contour found)
            return None
        buffer_zone = fill_zone  # For compatibility with rest of function

        # Create mask for buffer zone
        zone_mask = geometry_mask(
            [buffer_zone],
            out_shape=dem_data.shape,
            transform=transform,
            invert=True
        )

        # Valid data mask
        if dem_nodata is not None:
            valid_mask = (dem_data != dem_nodata) & zone_mask
        else:
            valid_mask = ~np.isnan(dem_data) & zone_mask

        # Exclude already-claimed areas
        if excluded_mask is not None:
            valid_mask = valid_mask & ~excluded_mask

        if not np.any(valid_mask):
            return None

        # Fill occurs where ground is below target AND above base (upslope of gabion)
        fill_mask = valid_mask & (dem_data < target_elev) & (dem_data >= base_elev - 1)

        if not np.any(fill_mask):
            return {
                'elevation': base_elev,
                'wall_length': line.length,
                'wall_height': wall_height,
                'fill_volume': 0,
                'flat_area': 0,
                'fill_efficiency': 0,
                'geometry': line,
                'fill_zone': fill_zone,
                'fill_mask': fill_mask
            }

        # Calculate fill
        fill_depths = target_elev - dem_data[fill_mask]
        fill_depths = np.clip(fill_depths, 0, wall_height)  # Can't fill more than wall height

        fill_volume = np.sum(fill_depths) * pixel_area
        flat_area = np.sum(fill_mask) * pixel_area

        # Calculate original slope in fill zone (for slope conversion metric)
        gy, gx = np.gradient(dem_data, abs(transform[4]), abs(transform[0]))
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)

        original_steep_cells = np.sum(fill_mask & (slope_deg > 15))
        steep_area_converted = original_steep_cells * pixel_area

        # Gabion rock volume
        rock_volume = line.length * wall_height * wall_width

        # Metrics
        fill_efficiency = flat_area / fill_volume if fill_volume > 0 else 0
        total_material = fill_volume + rock_volume

        return {
            'elevation': base_elev,
            'wall_length': line.length,
            'wall_height': wall_height,
            'fill_volume': fill_volume,
            'rock_volume': rock_volume,
            'total_material': total_material,
            'flat_area': flat_area,
            'steep_area_converted': steep_area_converted,
            'fill_efficiency': fill_efficiency,  # m² per m³ fill
            'material_efficiency': flat_area / total_material if total_material > 0 else 0,
            'geometry': line,
            'fill_zone': fill_zone,  # Store fill zone polygon for visualization
            'fill_mask': fill_mask  # Store for overlap prevention
        }


def optimize_gabion_placement(dem_path: str, wall_length: float, wall_height: float,
                               boundary_path: str = None, min_elev: float = None,
                               max_elev: float = None, wall_width: float = 0.5,
                               metric: str = 'fill_efficiency',
                               top_n: int = 5, min_spacing: float = None) -> list:
    """
    Find optimal gabion placement by evaluating all possible positions.

    Args:
        dem_path: Path to DEM raster
        wall_length: Desired gabion wall length in meters
        wall_height: Gabion wall height in meters
        boundary_path: Optional path to boundary polygon (gpkg/shp)
        min_elev, max_elev: Elevation range to consider
        metric: Optimization metric ('fill_efficiency', 'material_efficiency', 'flat_area')
        top_n: Number of top results to return
        min_spacing: Minimum distance between results (filters overlapping segments)

    Returns:
        List of top N placements sorted by metric (best first)
    """
    print(f"Extracting contours from DEM...")
    contours = extract_contours_from_dem(dem_path, interval=1.0,
                                          min_elev=min_elev, max_elev=max_elev)
    print(f"  Found {len(contours)} contour lines")

    # Load boundary if provided
    boundary = None
    if boundary_path:
        boundary = gpd.read_file(boundary_path)
        print(f"  Clipping to boundary: {boundary_path}")

    print(f"Extracting {wall_length}m segments...")
    segments = get_contour_segments(contours, wall_length, boundary)
    print(f"  Found {len(segments)} potential placements")

    if len(segments) == 0:
        print("  No valid segments found!")
        return []

    print(f"Evaluating placements (wall height: {wall_height}m)...")
    results = []
    for i, segment in enumerate(segments):
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(segments)}...")

        result = evaluate_gabion_placement(dem_path, segment, wall_height, wall_width)
        if result and result['fill_volume'] > 0:
            results.append(result)

    print(f"  {len(results)} placements with positive fill")

    # Sort by metric (descending)
    results.sort(key=lambda x: x.get(metric, 0), reverse=True)

    # Get DEM shape for tracking excluded areas
    with rasterio.open(dem_path) as dem:
        dem_shape = dem.read(1).shape

    # Greedy selection with overlap prevention
    filtered = []
    excluded_mask = np.zeros(dem_shape, dtype=bool)

    # Minimum elevation difference to prevent adjacent contour selection
    min_elev_diff = 5  # meters

    for result in results:
        result_elev = result['elevation']
        line = result['geometry']
        too_close = False

        for accepted in filtered:
            accepted_elev = accepted['elevation']
            elev_diff = abs(result_elev - accepted_elev)

            if elev_diff == 0:
                # Same contour - just check line distance
                if line.distance(accepted['geometry']) < min_spacing:
                    too_close = True
                    break
            elif elev_diff < min_elev_diff:
                # Adjacent contours (1-9m apart) - reject, they're too close geographically
                too_close = True
                break
            # else: different contours 10m+ apart - OK

        if too_close:
            continue

        # Check overlap with already-claimed areas
        fill_mask = result.get('fill_mask')
        if fill_mask is not None:
            overlap = np.sum(fill_mask & excluded_mask)
            total_fill = np.sum(fill_mask)
            if total_fill > 0:
                overlap_ratio = overlap / total_fill
                if overlap_ratio > 0.1:  # Skip if >10% overlap
                    continue

            # Claim this area
            excluded_mask = excluded_mask | fill_mask

        filtered.append(result)
        if len(filtered) >= top_n:
            break

    print(f"  {len(filtered)} after spacing + overlap filter")

    # Remove fill_mask from results (not needed in output, and can't serialize)
    for r in filtered:
        if 'fill_mask' in r:
            del r['fill_mask']

    return filtered


def export_results(results: list, output_path: str, crs):
    """Export top placements as GeoPackage with gabion lines and fill zones."""
    if not results:
        return

    # Export gabion lines
    gabion_data = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ['fill_zone', 'fill_mask']}
        gabion_data.append(row)

    gdf_gabions = gpd.GeoDataFrame(gabion_data, crs=crs)
    geoms = gdf_gabions['geometry']
    gdf_gabions = gdf_gabions.drop(columns=['geometry'])
    gdf_gabions['geometry'] = geoms
    gdf_gabions = gpd.GeoDataFrame(gdf_gabions, geometry='geometry', crs=crs)

    gdf_gabions.to_file(output_path, driver='GPKG', layer='gabions')
    print(f"Exported gabion lines: {output_path} (layer: gabions)")

    # Export fill zones as separate layer with full stats
    fill_zone_data = []
    for i, r in enumerate(results):
        if 'fill_zone' in r and r['fill_zone'] is not None:
            # Calculate costs
            wall_height = r['wall_height']
            wall_length = r['wall_length']
            fill_cost = r['fill_volume'] * 25
            rock_cost = r['rock_volume'] * 30
            mesh_area = wall_length * (2 * wall_height + 0.5)  # 0.5m wall width
            mesh_cost = mesh_area * 15
            total_cost = fill_cost + rock_cost + mesh_cost
            cost_per_m2 = total_cost / r['flat_area'] if r['flat_area'] > 0 else 0

            fill_zone_data.append({
                'rank': i + 1,
                'elevation': r['elevation'],
                'wall_height': wall_height,
                'wall_length': wall_length,
                'flat_area_m2': round(r['flat_area'], 1),
                'fill_vol_m3': round(r['fill_volume'], 1),
                'rock_vol_m3': round(r['rock_volume'], 1),
                'fill_effic': round(r['fill_efficiency'], 2),
                'fill_cost': round(fill_cost, 0),
                'rock_cost': round(rock_cost, 0),
                'mesh_cost': round(mesh_cost, 0),
                'total_cost': round(total_cost, 0),
                'cost_per_m2': round(cost_per_m2, 1),
                'geometry': r['fill_zone']
            })

    if fill_zone_data:
        gdf_zones = gpd.GeoDataFrame(fill_zone_data, crs=crs)
        gdf_zones.to_file(output_path, driver='GPKG', layer='fill_zones')
        print(f"Exported fill zones: {output_path} (layer: fill_zones)")


def calculate_slope(dem_data: np.ndarray, cell_size: float) -> np.ndarray:
    """Calculate slope in degrees from DEM array."""
    gy, gx = np.gradient(dem_data, cell_size, cell_size)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad)


def export_slope_raster(dem_path: str, output_path: str) -> str:
    """Export slope raster (in degrees) from a DEM."""
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        dem_nodata = dem.nodata
        cell_size = abs(dem.transform[0])
        profile = dem.profile.copy()

        # Calculate slope
        slope = calculate_slope(dem_data, cell_size)

        # Preserve nodata areas
        if dem_nodata is not None:
            slope[dem_data == dem_nodata] = -9999

        profile.update(dtype=np.float32, nodata=-9999)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope.astype(np.float32), 1)

    return output_path


def export_optimal_dem(dem_path: str, segment: dict, wall_height: float,
                       output_path: str, buffer_distance: float = None) -> str:
    """
    Export DEM modified by a single optimal gabion placement.

    Creates a modified DEM where the area behind (upslope of) the gabion
    is filled to the target elevation.

    Args:
        dem_path: Path to original DEM
        segment: Dict with 'geometry' (LineString) and 'elevation'
        wall_height: Gabion wall height in meters
        output_path: Output raster path
        buffer_distance: Distance to buffer the line for fill zone (default: wall_height * 5)

    Returns:
        Output path on success
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if buffer_distance is None:
        buffer_distance = wall_height * 5

    line = segment['geometry']
    base_elev = segment['elevation']
    target_elev = base_elev + wall_height

    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        dem_nodata = dem.nodata
        transform = dem.transform
        profile = dem.profile.copy()

        # Create upslope-only fill zone
        fill_zone = get_upslope_buffer(line, dem_data, transform, buffer_distance)

        # Create mask for the fill zone
        zone_mask = geometry_mask(
            [fill_zone],
            out_shape=dem_data.shape,
            transform=transform,
            invert=True  # True inside the zone
        )

        # Valid data mask
        if dem_nodata is not None:
            valid_mask = (dem_data != dem_nodata) & zone_mask
        else:
            valid_mask = ~np.isnan(dem_data) & zone_mask

        # Create modified DEM
        modified = dem_data.copy().astype(np.float32)

        # Fill where:
        # 1. Inside the upslope buffer zone (valid_mask)
        # 2. Original elevation is below target (needs fill)
        # 3. Original elevation is at or above base (upslope of gabion)
        fill_mask = valid_mask & (dem_data < target_elev) & (dem_data >= base_elev - 1)

        # Set filled areas to target elevation
        modified[fill_mask] = target_elev

        # Write output
        profile.update(dtype=np.float32)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(modified, 1)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Optimize gabion wall placement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  fill_efficiency     - m² flat area per m³ fill (default)
  material_efficiency - m² flat area per m³ total material (fill + rock)
  flat_area          - total flat area created (m²)
  steep_area_converted - steep land (>15°) converted to flat (m²)

Examples:
  %(prog)s --length 100 --height 1.0
  %(prog)s --length 100 --height 1.0 --metric flat_area
  %(prog)s --length 100 --height 0.5 1.0 1.5 --top 3
        """
    )
    parser.add_argument('--length', '-L', type=float, default=100,
                        help='Wall length in meters (default: 100)')
    parser.add_argument('--height', '-H', type=float, nargs='+', default=[1.0],
                        help='Wall height(s) in meters (default: 1.0)')
    parser.add_argument('--width', '-W', type=float, default=0.5,
                        help='Wall width in meters (default: 0.5)')
    parser.add_argument('--min-elev', type=float, default=1070,
                        help='Minimum elevation to consider (default: 1070)')
    parser.add_argument('--max-elev', type=float, default=1110,
                        help='Maximum elevation to consider (default: 1110)')
    parser.add_argument('--metric', '-m', type=str, default='fill_efficiency',
                        choices=['fill_efficiency', 'material_efficiency', 'flat_area', 'steep_area_converted'],
                        help='Optimization metric (default: fill_efficiency)')
    parser.add_argument('--top', '-n', type=int, default=5,
                        help='Number of top results to show (default: 5)')
    parser.add_argument('--min-spacing', '-s', type=float, default=None,
                        help='Minimum distance between results in meters (filters overlapping segments)')
    parser.add_argument('--export', '-e', action='store_true',
                        help='Export top placements as GeoPackage')
    parser.add_argument('--export-dem', action='store_true',
                        help='Export modified DEM for each top placement')

    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent
    dem_path = project_dir.parent / "reprojected.tif"  # DEM is in parent directory
    boundary_path = project_dir.parent / "terraces_1m.gpkg"  # Use terraces as rough boundary

    # Get CRS from DEM
    with rasterio.open(dem_path) as dem:
        crs = dem.crs

    print("=" * 70)
    print("GABION PLACEMENT OPTIMIZER")
    print("=" * 70)
    print()
    print(f"Parameters:")
    print(f"  Wall length: {args.length} m")
    print(f"  Wall heights: {args.height} m")
    print(f"  Elevation range: {args.min_elev} - {args.max_elev} m")
    print(f"  Optimization metric: {args.metric}")
    print()

    all_results = {}

    for height in args.height:
        print("-" * 70)
        print(f"WALL HEIGHT: {height} m")
        print("-" * 70)

        results = optimize_gabion_placement(
            dem_path=str(dem_path),
            wall_length=args.length,
            wall_height=height,
            boundary_path=None,  # Search full DEM for now
            min_elev=args.min_elev,
            max_elev=args.max_elev,
            wall_width=args.width,
            metric=args.metric,
            top_n=args.top,
            min_spacing=args.min_spacing
        )

        all_results[height] = results

        if results:
            print()
            print(f"TOP {len(results)} PLACEMENTS (by {args.metric}):")
            print("-" * 70)
            print(f"{'Rank':<5} {'Elev':<8} {'Fill m³':<10} {'Flat m²':<10} {'Efficiency':<12} {'Cost est.':<12}")
            print("-" * 70)

            for i, r in enumerate(results):
                fill_cost = r['fill_volume'] * 25
                rock_cost = r['rock_volume'] * 30
                mesh_cost = r['wall_length'] * (2 * height + args.width) * 15
                total_cost = fill_cost + rock_cost + mesh_cost

                print(f"{i+1:<5} {r['elevation']:<8.0f} {r['fill_volume']:<10.0f} "
                      f"{r['flat_area']:<10.0f} {r['fill_efficiency']:<12.2f} "
                      f"{total_cost:<10,.0f} лв")

            # Detailed cost breakdown
            print()
            print("COST BREAKDOWN:")
            print("-" * 70)
            for i, r in enumerate(results):
                fill_cost = r['fill_volume'] * 25
                rock_cost = r['rock_volume'] * 30
                mesh_area = r['wall_length'] * (2 * height + args.width)
                mesh_cost = mesh_area * 15
                total_cost = fill_cost + rock_cost + mesh_cost

                print(f"Rank {i+1} (elev {r['elevation']:.0f}m):")
                print(f"  Fill:  {r['fill_volume']:>6.0f} m³ × 25 лв/m³ = {fill_cost:>10,.0f} лв")
                print(f"  Rock:  {r['rock_volume']:>6.0f} m³ × 30 лв/m³ = {rock_cost:>10,.0f} лв")
                print(f"  Mesh:  {mesh_area:>6.0f} m² × 15 лв/m² = {mesh_cost:>10,.0f} лв")
                print(f"  TOTAL: {' ':>19} {total_cost:>10,.0f} лв")
                print(f"  Flat area created: {r['flat_area']:.0f} m² → {total_cost/r['flat_area']:.0f} лв/m²")
                print()

            # Export GeoPackage if requested
            if args.export:
                output_path = project_dir / f"optimal_gabions_{height}m_{args.metric}.gpkg"
                export_results(results, str(output_path), crs)

            # Export modified DEMs if requested
            if args.export_dem:
                print()
                print("Exporting modified DEMs and slope rasters...")
                for rank, result in enumerate(results, 1):
                    dem_output = project_dir / f"optimal_dem_{height}m_{args.metric}_rank{rank}.tif"
                    slope_output = project_dir / f"optimal_slope_{height}m_{args.metric}_rank{rank}.tif"
                    export_optimal_dem(
                        str(dem_path),
                        result,
                        height,
                        str(dem_output)
                    )
                    export_slope_raster(str(dem_output), str(slope_output))
                    print(f"  Exported: {dem_output}")
                    print(f"  Exported: {slope_output}")

    # Summary comparison if multiple heights
    if len(args.height) > 1:
        print()
        print("=" * 70)
        print("BEST PLACEMENT PER HEIGHT")
        print("=" * 70)
        print()
        print(f"{'Height':<10} {'Elevation':<10} {'Fill m³':<10} {'Flat m²':<10} {'Efficiency':<12}")
        print("-" * 52)
        for height, results in all_results.items():
            if results:
                r = results[0]
                print(f"{height:<10.1f} {r['elevation']:<10.0f} {r['fill_volume']:<10.0f} "
                      f"{r['flat_area']:<10.0f} {r['fill_efficiency']:<12.2f}")
        print()

    # Print validation instructions if DEMs were exported
    if args.export_dem:
        print()
        print("=" * 70)
        print("VALIDATION WORKFLOW IN QGIS")
        print("=" * 70)
        print()
        print("1. Load an exported DEM (e.g., optimal_dem_1.0m_fill_efficiency_rank1.tif)")
        print("2. Raster → Analysis → Slope")
        print("3. Apply permaculture slope styling:")
        print("   - 0-8°: green (optimal for cultivation)")
        print("   - 8-15°: light green (suitable with care)")
        print("   - 15-25°: yellow (marginal)")
        print("   - >25°: red (too steep)")
        print("4. Compare to original slope map")
        print("5. Overlay optimal_gabions_*.gpkg to see gabion positions")
        print()


if __name__ == "__main__":
    main()
