# Gabion Terrace Optimization - Knowledge Base

## Tools Developed

### 1. `optimize_gabion_placement.py`
Finds optimal gabion locations along contour lines.

**Key features:**
- Extracts contours from DEM at 1m intervals
- Evaluates segments of specified length (default 100m)
- **Upslope-only buffers** - only counts fill area on the uphill side of gabion
- **Overlap prevention** - tracks claimed pixels, rejects >10% overlap
- **Spacing filters:**
  - Line-to-line distance (not just centroids)
  - Minimum elevation difference (10m) to prevent adjacent contour selection
  - Same contour allowed if 50m+ apart

**Usage:**
```bash
python optimize_gabion_placement.py --height 1.0 --top 5 --min-spacing 50 --export
```

### 2. `optimize_locations_then_height.py`
Two-stage optimization: find locations first, then optimize height per location.

**Usage:**
```bash
python optimize_locations_then_height.py --budget 100000 --metric flat_area --export
```

**Metrics:**
- `fill_efficiency` - m² per m³ fill (favors short walls, minimal fill)
- `flat_area` - absolute area created (favors tall walls)
- `material_efficiency` - m² per m³ total material

### 3. `build_100k_project.py`
Combines gabions from multiple height optimizations into a single project within budget.

**Has cross-height filtering** to prevent gabions at adjacent elevations (e.g., 0.5m wall at 1087m blocking 1.0m wall at 1088m).

### 4. `build_combined_dem.py`
Creates combined DEM and slope raster showing terrain after ALL gabions are built.

### 5. `calculate_fill_volume.py`
Calculates fill volumes for user-drawn gabion placements (original tool, uses terrace polygons).

---

## What Works

1. **Contour extraction** from DEM using matplotlib - reliable
2. **Segment extraction** along contours with overlap (step = length/4)
3. **Line-to-line distance** for spacing - better than centroid distance
4. **Elevation difference filter** (10m minimum) - prevents adjacent contour selection
5. **Upslope buffer detection** - compares average elevation on left vs right side of line
6. **Greedy selection** with pixel-level overlap tracking
7. **Cost estimation**: fill (25 лв/m³), rock (30 лв/m³), mesh (15 лв/m²)

---

## What Doesn't Work / Limitations

### 1. "Flat Area" Metric is Misleading
The optimizer reports "flat_area" as pixels where fill occurs. But this does NOT equal actual 0-8° slope land created.

**Example from 100k project:**
- Optimizer reported: 888 m² flat area
- Actual 0-8° slope gain: **236 m²** (only 27% of reported)

**Why:** Edge effects. Each gabion creates artificial "cliffs" at the wall that show as >25° in slope calculations. Narrow terraces have proportionally more edge than center.

### 2. Upslope Buffer is Geometric, Not Hydrological
Current approach: `line.buffer(height * 5)` then filter to upslope side by elevation.

**Problem:** Real fill would follow watershed boundaries, not circular buffers. Some "upslope" area may be across a ridge and wouldn't actually fill.

**Proper solution:** Use flow direction analysis (D8 algorithm) to find true catchment area.

### 3. Short Walls Create Almost No Usable Area
0.5m walls with upslope-only buffer create 8-28 m² per 100m gabion. Not cost-effective.

**Best results:** 1.5m walls create 260-340 m² per gabion.

### 4. Limited Locations with Proper Spacing
With 10m elevation difference + 50m line spacing, only **3 locations** found in the study area (1070-1110m range).

---

## Key Questions Answered

### Q: Where are the best spots for gabion dams?
**A:** Use `optimize_gabion_placement.py` with `--metric flat_area` and `--min-spacing 50`. The tool finds contour segments that maximize flat area creation. In the Sabazii site, best locations are at 1077m, 1088m, and 1102m elevations.

### Q: What wall height is optimal?
**A:** Depends on metric:
- **fill_efficiency**: shorter walls (0.5m) - minimal material but tiny areas
- **flat_area**: taller walls (1.5m) - more absolute area, better cost/m²

For maximizing usable land, **1.5m walls** at 37 лв/m² beat 0.5m walls at 100+ лв/m².

### Q: How much flat land can 100k лв create?
**A:** With honest accounting (upslope-only, no overlap):
- **Optimizer estimate:** 888 m² (3 gabions × 1.5m height)
- **Actual slope improvement:** ~236 m² of new 0-8° land
- **Cost:** 32,505 лв spent, 67,495 лв reserve
- **Real cost:** ~138 лв/m² of actual optimal-slope land

### Q: Why do gabions overlap in QGIS?
**A:** Three issues fixed:
1. Centroid distance vs line distance - **fixed**: now uses `line.distance()`
2. Adjacent contours (1m apart) are geographically close - **fixed**: 10m elevation difference minimum
3. Cross-height conflicts (0.5m at 1087m vs 1.0m at 1088m) - **fixed** in `build_100k_project.py`

### Q: Is gabion terracing cost-effective for flat land creation?
**A:** **No, not purely for cultivation area.** At 138+ лв/m², it's far more expensive than buying agricultural land (1-5 лв/m²).

**But gabions provide other value:**
- Water retention and infiltration
- Erosion control
- Long-term soil building
- Microclimate modification

---

## Current Best Results

**3 gabions at 1.5m height:**
| Location | Flat Area | Cost |
|----------|-----------|------|
| 1088m | 340 m² | 11,541 лв |
| 1102m | 288 m² | 10,076 лв |
| 1077m | 260 m² | 10,888 лв |
| **TOTAL** | **888 m²** (reported) / **236 m²** (actual slope gain) | **32,505 лв** |

---

## Files in 100k_project/

```
Scripts:
- optimize_gabion_placement.py      # Main optimizer
- optimize_locations_then_height.py # Two-stage optimizer
- build_100k_project.py             # Combine heights within budget
- build_combined_dem.py             # Generate combined DEM/slope
- calculate_fill_volume.py          # Volume calculator for drawn gabions

Output (current best):
- optimal_project_flat_area.gpkg    # 3 gabions, 1.5m height
- combined_slope_optimal.tif        # Slope after construction
- combined_dem_optimal.tif          # DEM after construction

Old outputs (may be stale):
- optimal_gabions_*.gpkg            # Single-height optimizations
- project_100k_gabions.gpkg         # Old combined project
```

---

## Next Steps to Investigate

1. **Longer walls** - Try 150m or 200m segments for more area per gabion
2. **Different elevation ranges** - Search outside 1070-1110m
3. **Reduced spacing** - Accept 30m spacing to fit more gabions
4. **Alternative metrics** - Optimize for water retention, not just flat area
5. **Hydrological analysis** - Use flow accumulation to find natural water collection points
6. **Staged construction** - Build highest-value gabion first, measure results, then iterate

---

## Cost Assumptions

| Material | Unit Cost | Notes |
|----------|-----------|-------|
| Fill material | 25 лв/m³ | Local soil/aggregate |
| Gabion rock | 30 лв/m³ | 10-15cm stones |
| Gabion mesh | 15 лв/m² | Galvanized wire mesh |

**Labor not included** - costs are materials only.
