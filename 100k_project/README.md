# 100K Gabion Terrace Project

Optimal gabion dam placement for 100,000 лв budget.

## Project Summary

| Metric | Value |
|--------|-------|
| Budget | 100,000 лв |
| Spent | 94,858 лв |
| Reserve | 5,142 лв |
| Total gabions | 10 |
| Total flat area | 4,552 m² (~0.46 ha) |
| Average cost | 21 лв/m² |

## Phases

### Phase 1: 0.5m Walls (7 gabions)
Best cost efficiency at 16-20 лв/m².

| # | Elevation | Flat Area | Cost |
|---|-----------|-----------|------|
| 1 | 1095m | 392 m² | 6,144 лв |
| 2 | 1095m | 420 m² | 6,528 лв |
| 3 | 1105m | 392 m² | 6,397 лв |
| 4 | 1075m | 316 m² | 5,753 лв |
| 5 | 1076m | 296 m² | 5,598 лв |
| 6 | 1070m | 276 m² | 5,487 лв |
| 7 | 1073m | 328 m² | 6,053 лв |

**Subtotal Phase 1:** 41,960 лв → 2,420 m²

### Phase 2: 1.0m Walls (3 gabions)
Higher walls for remaining budget, 24-25 лв/m².

| # | Elevation | Flat Area | Cost |
|---|-----------|-----------|------|
| 8 | 1088m | 576 m² | 14,569 лв |
| 9 | 1105m | 824 m² | 19,903 лв |
| 10 | 1070m | 732 m² | 18,426 лв |

**Subtotal Phase 2:** 52,898 лв → 2,132 m²

## Files

### Scripts
- `optimize_gabion_placement.py` - Find optimal gabion locations
- `calculate_fill_volume.py` - Calculate volumes for drawn gabions
- `build_100k_project.py` - Build combined project layer

### Output Layers (GeoPackage)
- `project_100k_gabions.gpkg` - **Combined project layer (10 gabions)**
- `optimal_gabions_0.5m_fill_efficiency.gpkg` - Phase 1 candidates
- `optimal_gabions_1.0m_fill_efficiency.gpkg` - Phase 2 candidates

### Rasters (GeoTIFF)
- `optimal_dem_0.5m_fill_efficiency_rank*.tif` - Modified DEMs per gabion
- `optimal_slope_0.5m_fill_efficiency_rank*.tif` - Slope maps per gabion

## Visualization in QGIS

1. Load `project_100k_gabions.gpkg`
2. Style by `phase` attribute (Phase 1 = green, Phase 2 = blue)
3. Label with `project_rank` or `elevation`
4. Load slope rasters to see terrain effect

### Slope Styling (Permaculture)
| Slope | Color | Use |
|-------|-------|-----|
| 0-8° | Green | Optimal cultivation |
| 8-15° | Light green | Suitable with care |
| 15-25° | Yellow | Marginal |
| >25° | Red | Too steep |

## Regenerate Project

```bash
# From project root directory:

# Step 1: Generate optimal placements
.venv/bin/python optimize_gabion_placement.py \
    --height 0.5 --top 10 --min-spacing 50 --export

.venv/bin/python optimize_gabion_placement.py \
    --height 1.0 --top 10 --min-spacing 50 --export

# Step 2: Build combined layer
cd 100k_project
../.venv/bin/python build_100k_project.py
```

## Cost Assumptions

| Material | Unit Cost |
|----------|-----------|
| Fill material | 25 лв/m³ |
| Gabion rock | 30 лв/m³ |
| Gabion mesh | 15 лв/m² |

## Caveats

1. **Simplified hydrology** - Fill zones use geometric buffers, not true watershed analysis
2. **No site constraints** - Doesn't account for access roads, soil stability, existing features
3. **Overlapping locations** - Some Phase 1 and Phase 2 gabions may be close; verify in QGIS
4. **Labor not included** - Costs are materials only
