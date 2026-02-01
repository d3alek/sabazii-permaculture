# Gabion Terrace Optimization - Knowledge Base

## Tools Developed

### 1. `optimize_gabion_placement.py`
Finds optimal gabion locations along contour lines using **contour-based fill zones**.

**Key features:**
- Extracts contours from DEM at 1m intervals
- Evaluates segments of specified length (default 100m)
- **Contour-based fill zones** - bounded by gabion line and uphill contour at target elevation
- **Overlap prevention** - tracks claimed pixels, rejects >10% overlap
- **Spacing filters:**
  - Minimum elevation difference (5m) between gabions on different contours
  - Same contour allowed if 50m+ apart (--min-spacing)
- **Exports fill zone polygons** with full stats as attributes

**Usage:**
```bash
python optimize_gabion_placement.py --height 1.0 --top 10 --min-spacing 50 --export
python optimize_gabion_placement.py --height 0.5 1.0 1.5 2.0 2.5 3.0 --metric flat_area --top 15 --export
```

**Output:** GeoPackage with two layers:
- `gabions` - gabion line geometries
- `fill_zones` - terrace polygons with attributes:
  - `flat_area_m2`, `fill_vol_m3`, `rock_vol_m3`
  - `fill_cost`, `rock_cost`, `mesh_cost`, `total_cost`
  - `cost_per_m2`, `fill_effic`

### 2. `optimize_locations_then_height.py`
Two-stage optimization: find locations first, then optimize height per location.

### 3. `build_100k_project.py`
Combines gabions from multiple height optimizations into a single project within budget.

### 4. `build_combined_dem.py`
Creates combined DEM and slope raster showing terrain after ALL gabions are built.

### 5. `calculate_fill_volume.py`
Calculates fill volumes for user-drawn gabion placements (original tool, uses terrace polygons).

---

## Algorithm: Contour-Based Fill Zones

The fill zone for each gabion is calculated as follows:

1. **Gabion line** at base elevation (e.g., 1094m)
2. **Target contour** at base + wall_height (e.g., 1095.5m for 1.5m wall)
3. **Lateral edges** connect gabion endpoints to closest points on target contour
4. **Fill polygon** = gabion line + target contour segment + lateral edges

This replaces the old geometric buffer approach (`line.buffer(height * 5)`) which:
- Didn't follow actual terrain
- Could include area across ridges that wouldn't fill
- Over/underestimated terrace sizes

---

## Current Best Results (100k лв Budget)

### Cost per m² by Wall Height

| Height | Gabions | Total Area | Total Cost | Cost/m² | Fits 100k? |
|--------|---------|------------|------------|---------|------------|
| 0.5m   | 8       | 812 m²     | 27,459 лв  | 34 лв   | ✓ (72k spare) |
| **1.0m** | **8** | **2,336 m²** | **64,951 лв** | **28 лв** | **✓ (35k spare)** |
| 1.5m   | 9       | 4,148 m²   | 131,930 лв | 32 лв   | ✗ (32k over) |
| 2.0m   | 9       | 5,716 m²   | 207,283 лв | 36 лв   | ✗ |
| 2.5m   | 9       | 6,972 m²   | 290,049 лв | 42 лв   | ✗ |
| 3.0m   | 10      | 9,152 m²   | 441,760 лв | 48 лв   | ✗ |
| 3.5m   | 10      | 10,836 m²  | 586,687 лв | 54 лв   | ✗ |
| 4.0m   | 9       | 10,868 m²  | 665,724 лв | 61 лв   | ✗ |
| 4.5m   | 10      | 13,508 m²  | 899,296 лв | 67 лв   | ✗ |
| 5.0m   | 10      | 14,860 m²  | 1,091,495 лв | 73 лв | ✗ |

### Best Single Gabion per Height

| Height | Elevation | Flat Area | Fill m³ | Cost | Cost/m² |
|--------|-----------|-----------|---------|------|---------|
| 0.5m   | 1095m     | 220 m²    | 33      | 3,834 лв | 17 лв |
| 1.0m   | 1095m     | 440 m²    | 198     | 10,207 лв | 23 лв |
| 1.5m   | 1094m     | 680 m²    | 393     | 17,335 лв | 25 лв |
| 2.0m   | 1098m     | 908 m²    | 819     | 30,218 лв | 33 лв |
| 2.5m   | 1097m     | 1,160 m²  | 1,246   | 43,146 лв | 37 лв |
| 3.0m   | 1097m     | 1,400 m²  | 1,884   | 61,349 лв | 44 лв |
| 3.5m   | 1097m     | 1,576 m²  | 2,630   | 82,259 лв | 52 лв |
| 4.0m   | 1096m     | 1,756 m²  | 2,934   | 92,095 лв | 52 лв |
| 4.5m   | 1096m     | 1,936 m²  | 3,860   | 117,493 лв | 61 лв |
| 5.0m   | 1095m     | 2,184 m²  | 5,064   | 149,847 лв | 69 лв |

### Recommendation for 100k лв

**1.0m walls**: 8 gabions, **2,336 m²** total area, 64,951 лв (35k лв spare)

---

## Configuration

### Spacing Parameters (in `optimize_gabion_placement.py`)

```python
min_elev_diff = 5  # meters - minimum elevation between gabions on different contours
```

Command line:
- `--min-spacing 50` - minimum distance between gabions on same contour (meters)
- `--top N` - number of gabions to return

### Cost Assumptions

| Material | Unit Cost | Notes |
|----------|-----------|-------|
| Fill material | 25 лв/m³ | Local soil/aggregate |
| Gabion rock | 30 лв/m³ | 10-15cm stones |
| Gabion mesh | 15 лв/m² | Galvanized wire mesh |

**Labor not included** - costs are materials only.

---

## Files in 100k_project/

```
Scripts:
- optimize_gabion_placement.py      # Main optimizer (contour-based)
- optimize_locations_then_height.py # Two-stage optimizer
- build_100k_project.py             # Combine heights within budget
- build_combined_dem.py             # Generate combined DEM/slope
- calculate_fill_volume.py          # Volume calculator for drawn gabions

Output (current):
- optimal_gabions_0.5m_flat_area.gpkg  # Results for each height
- optimal_gabions_1.0m_flat_area.gpkg  # with gabions + fill_zones layers
- optimal_gabions_1.5m_flat_area.gpkg
- ... up to 5.0m
```

---

## Key Learnings

### 1. Contour-Based Fill Zones are Essential
Geometric buffers (`line.buffer(height * 5)`) don't follow terrain. The new approach:
- Uses actual uphill contour as boundary
- Connects gabion endpoints to closest points on target contour
- Produces accurate terrace polygons for visualization

### 2. Cost/m² Increases with Height
- 0.5m walls: 17 лв/m² (most efficient)
- 3.0m walls: 44 лв/m² (2.6x more expensive per m²)
- But taller walls create more area per gabion

### 3. Diminishing Returns Above 2.0m
Fill volume grows faster than area for taller walls:
- 1.0m: 198 m³ fill → 440 m² (2.2 efficiency)
- 3.0m: 1,884 m³ fill → 1,400 m² (0.74 efficiency)

### 4. Budget Determines Optimal Height
- **Maximize area within budget**: Use shorter walls (0.5-1.0m)
- **Maximize area per gabion**: Use taller walls (2.5-3.0m)
- **100k лв budget**: 1.0m walls are optimal

---

## Next Steps to Investigate

1. **Mixed heights** - Use 1.0m walls at best locations, 0.5m elsewhere
2. **Longer walls** - Try 150m or 200m segments
3. **Different elevation ranges** - Search outside 1070-1110m
4. **Water retention metric** - Optimize for catchment, not just flat area
5. **Staged construction** - Build highest-value gabion first, measure, iterate
