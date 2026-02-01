#!/usr/bin/env python3
"""
Build combined DEM and slope raster showing all gabions from the project.

Creates a single modified DEM with all gabion terraces filled,
then calculates the resulting slope map.

Usage:
  python build_combined_dem.py
"""

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
from pathlib import Path


def calculate_slope(dem_data: np.ndarray, cell_size: float) -> np.ndarray:
    """Calculate slope in degrees from DEM array."""
    gy, gx = np.gradient(dem_data, cell_size, cell_size)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad)


def get_upslope_buffer(line, dem_data: np.ndarray, transform,
                       buffer_dist: float):
    """
    Create a buffer on the upslope side of the gabion line only.
    """
    from shapely.ops import unary_union

    try:
        left_offset = line.parallel_offset(buffer_dist, side='left')
        right_offset = line.parallel_offset(buffer_dist, side='right')
    except Exception:
        return line.buffer(buffer_dist)

    if left_offset.is_empty or right_offset.is_empty:
        return line.buffer(buffer_dist)

    def make_side_polygon(offset_line, side):
        try:
            poly = line.buffer(buffer_dist, single_sided=True)
            if side == 'right':
                poly = line.buffer(-buffer_dist, single_sided=True)
            if poly.is_valid and not poly.is_empty:
                return poly
        except Exception:
            pass

        try:
            if offset_line.geom_type == 'MultiLineString':
                offset_line = max(offset_line.geoms, key=lambda g: g.length)
            if offset_line.is_empty:
                return None
            coords = list(line.coords) + list(reversed(list(offset_line.coords)))
            if len(coords) >= 4:
                poly = Polygon(coords)
                if poly.is_valid:
                    return poly
        except Exception:
            pass
        return None

    left_poly = make_side_polygon(left_offset, 'left')
    right_poly = make_side_polygon(right_offset, 'right')

    if left_poly is None and right_poly is None:
        return line.buffer(buffer_dist)

    def get_avg_elevation(poly):
        if poly is None or not poly.is_valid:
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

    if left_elev > right_elev and left_poly is not None:
        return left_poly
    elif right_poly is not None:
        return right_poly
    else:
        return line.buffer(buffer_dist)


def apply_gabion_fill(dem_data: np.ndarray, transform, nodata,
                      geometry, base_elev: float, wall_height: float,
                      buffer_distance: float = None) -> np.ndarray:
    """
    Apply fill from a single gabion to the DEM.

    Uses upslope-only buffer to avoid filling downslope areas.
    Modifies dem_data in place.
    """
    if buffer_distance is None:
        buffer_distance = wall_height * 5

    target_elev = base_elev + wall_height

    # Create upslope-only fill zone
    fill_zone = get_upslope_buffer(geometry, dem_data, transform, buffer_distance)

    # Create mask for the fill zone
    zone_mask = geometry_mask(
        [fill_zone],
        out_shape=dem_data.shape,
        transform=transform,
        invert=True
    )

    # Valid data mask
    if nodata is not None:
        valid_mask = (dem_data != nodata) & zone_mask
    else:
        valid_mask = ~np.isnan(dem_data) & zone_mask

    # Fill where:
    # 1. Inside the upslope buffer zone
    # 2. Original elevation is below target
    # 3. Original elevation is at or above base (upslope of gabion)
    fill_mask = valid_mask & (dem_data < target_elev) & (dem_data >= base_elev - 1)

    # Apply fill
    dem_data[fill_mask] = target_elev

    return dem_data


def build_combined_dem(original_dem_path: str, project_gpkg_path: str,
                       output_dem_path: str, output_slope_path: str):
    """
    Build combined DEM with all gabions applied.

    Args:
        original_dem_path: Path to original DEM
        project_gpkg_path: Path to project GeoPackage with all gabions
        output_dem_path: Output path for modified DEM
        output_slope_path: Output path for slope raster
    """
    # Load project gabions
    gabions = gpd.read_file(project_gpkg_path)
    print(f"Loaded {len(gabions)} gabions from project")

    # Load and modify DEM
    with rasterio.open(original_dem_path) as dem:
        dem_data = dem.read(1).astype(np.float32)
        nodata = dem.nodata
        transform = dem.transform
        cell_size = abs(transform[0])
        profile = dem.profile.copy()

        # Ensure gabions are in same CRS
        if gabions.crs != dem.crs:
            gabions = gabions.to_crs(dem.crs)

        # Apply each gabion
        for idx, row in gabions.iterrows():
            print(f"  Applying gabion #{row['project_rank']}: "
                  f"{row['wall_height_m']}m @ {row['elevation']:.0f}m")

            apply_gabion_fill(
                dem_data,
                transform,
                nodata,
                row.geometry,
                base_elev=row['elevation'],
                wall_height=row['wall_height_m']
            )

    # Calculate slope
    print("Calculating slope...")
    slope = calculate_slope(dem_data, cell_size)

    # Preserve nodata areas
    if nodata is not None:
        slope[dem_data == nodata] = -9999

    # Write modified DEM
    profile.update(dtype=np.float32)
    with rasterio.open(output_dem_path, 'w', **profile) as dst:
        dst.write(dem_data, 1)
    print(f"Exported: {output_dem_path}")

    # Write slope
    profile.update(dtype=np.float32, nodata=-9999)
    with rasterio.open(output_slope_path, 'w', **profile) as dst:
        dst.write(slope.astype(np.float32), 1)
    print(f"Exported: {output_slope_path}")

    # Print stats comparison
    with rasterio.open(original_dem_path) as orig:
        orig_data = orig.read(1)
        orig_slope = calculate_slope(orig_data, cell_size)

        if nodata is not None:
            valid = orig_data != nodata
        else:
            valid = ~np.isnan(orig_data)

        # Count cells by slope category
        def count_slope_categories(s, mask):
            flat = np.sum(mask & (s <= 8))
            gentle = np.sum(mask & (s > 8) & (s <= 15))
            moderate = np.sum(mask & (s > 15) & (s <= 25))
            steep = np.sum(mask & (s > 25))
            return flat, gentle, moderate, steep

        orig_cats = count_slope_categories(orig_slope, valid)
        new_cats = count_slope_categories(slope, valid)

        cell_area = cell_size * cell_size

        print()
        print("=" * 60)
        print("SLOPE COMPARISON (before vs after)")
        print("=" * 60)
        print(f"{'Category':<20} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-" * 60)

        labels = ['0-8° (optimal)', '8-15° (gentle)', '15-25° (marginal)', '>25° (steep)']
        for i, label in enumerate(labels):
            before = orig_cats[i] * cell_area
            after = new_cats[i] * cell_area
            change = after - before
            sign = '+' if change >= 0 else ''
            print(f"{label:<20} {before:>10,.0f} m²  {after:>10,.0f} m²  {sign}{change:>10,.0f} m²")

        print()


def main():
    project_dir = Path(__file__).parent
    parent_dir = project_dir.parent

    original_dem = parent_dir / "reprojected.tif"
    project_gpkg = project_dir / "project_100k_gabions.gpkg"
    output_dem = project_dir / "combined_dem_100k.tif"
    output_slope = project_dir / "combined_slope_100k.tif"

    if not original_dem.exists():
        print(f"Missing: {original_dem}")
        return

    if not project_gpkg.exists():
        print(f"Missing: {project_gpkg}")
        print("Run build_100k_project.py first")
        return

    build_combined_dem(
        str(original_dem),
        str(project_gpkg),
        str(output_dem),
        str(output_slope)
    )


if __name__ == "__main__":
    main()
