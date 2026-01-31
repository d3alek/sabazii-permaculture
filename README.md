# Sabazi Osnovno Kare - LCC to Contours Workflow

## Source Data

- **File**: `Sabazi_Osnovno kare.lcc` (XGrids 3D Gaussian Splatting format)
- **Total splats**: 97,842,416 (across 7 LOD levels)
- **CRS**: EPSG:32635 (UTM zone 35N)
- **Offset**: [331551.35772697697, 4634342.3726356402, 1096.0618000105001]

## Conversion Process

### Tool Used
- **splat-transform** v1.2.0 ([PlayCanvas](https://github.com/playcanvas/splat-transform))
- Install: `npm install -g @playcanvas/splat-transform`

### Required Input Files
The LCC format requires these files to be present:
- `Sabazi_Osnovno kare.lcc` (metadata, 1.9KB)
- `data.bin` (position data, 3GB)
- `shcoef.bin` (spherical harmonics, 5.9GB)
- `index.bin` (index, 24KB)

### Command
```bash
# Convert LOD level 6 only (lowest detail, 748K points)
# Full dataset (98M points) causes OOM on 16GB RAM
splat-transform -O 6 "Sabazi_Osnovno kare.lcc" output_lod6.csv
```

## Output Files

| File | Description |
|------|-------------|
| `output_lod6.ply` | PLY format, local coordinates, 180MB |
| `output_lod6.csv` | CSV with all Gaussian properties, local coordinates |
| `georeferenced_xyz.csv` | XYZ only, UTM coordinates (EPSG:32635), 748,037 points |

### Georeferencing
The offset from LCC metadata was applied:
```
X_utm = X_local + 331551.35772697697
Y_utm = Y_local + 4634342.3726356402
Z_utm = Z_local + 1096.0618000105001
```

## Vegetation Filtering (CSF)

Photogrammetry captures canopy surface, not ground. Use Cloth Simulation Filter to extract ground points before creating the DEM.

### Install dependencies
```bash
python -m venv .venv
.venv/bin/pip install cloth-simulation-filter numpy
```

### Run filter
```bash
.venv/bin/python csf_filter.py
```

The `csf_filter.py` script:
```python
import numpy as np
import CSF

# Load points
data = np.loadtxt('georeferenced_xyz.csv', delimiter=',', skiprows=1)

# Configure CSF
csf = CSF.CSF()
csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 1.0  # meters
csf.params.rigidness = 2           # 1=flat, 2=gentle slope, 3=steep
csf.params.class_threshold = 0.5   # lower = stricter
csf.params.interations = 500

# Run filter
csf.setPointCloud(data)
ground_idx = CSF.VecInt()
offground_idx = CSF.VecInt()
csf.do_filtering(ground_idx, offground_idx)

# Save ground points
ground_points = data[list(ground_idx)]
np.savetxt('ground_points.csv', ground_points, delimiter=',',
           header='x,y,z', comments='', fmt='%.6f')
```

### Output
- `ground_points.csv` - filtered ground points (vegetation removed)
- Typical result: ~24% classified as ground, ~76% removed as vegetation

### Tuning
- Too many vegetation points in ground → lower `class_threshold` (try 0.3)
- Ground too sparse/holes → raise `class_threshold` (try 0.7)
- Steep terrain → set `rigidness = 3`

## QGIS Workflow

### Step 1: Import filtered point cloud
1. **Layer → Add Layer → Add Delimited Text Layer**
2. File: `ground_points.csv` (CSF-filtered, not the original)
3. X field: `x`, Y field: `y`, Z field: `z`
4. Geometry CRS: `EPSG:32635`

### Step 2: Create DEM (TIN interpolation)
1. **Processing → Toolbox** → search "TIN interpolation"
2. Input layer: ground_points
3. Interpolation attribute: `z`
4. Pixel size: `1` (1 meter)
5. Run

### Step 3: Smooth the DEM
Light smoothing to reduce remaining noise:

1. **Processing Toolbox** → search "Warp"
2. Select **GDAL → Warp (reproject)**
3. Input: TIN interpolation raster
4. Output resolution: `3` or `4` (meters)
5. Resampling method: `median` (robust to outliers)
6. Run

### Step 4: Generate contours
1. **Raster → Extraction → Contour**
2. Input layer: the smoothed/warped raster
3. Interval: `1` (1 meter contours)
4. Run

### Step 5: Style contours
Use rule-based styling for major/minor contours:
- Major contours (every 5m): `"ELEV" % 5 = 0` → color `#5c4a3a`, weight 0.5
- Minor contours: color `#c4a882`, weight 0.25

Enable labels on major contours:
- Rule-based labeling with same filter
- Placement: Curved, On line
- Enable buffer for readability

## Terrain Analysis

### Slope Map
1. **Raster → Analysis → Slope**
2. Input: smoothed DEM
3. Output in degrees

Style with 5 classes for permaculture planning:

| Class | Label | HTML Color |
|-------|-------|------------|
| 0-8° | Ideal | `#1a9850` |
| 8-15° | Good | `#91cf60` |
| 15-22° | Terracing needed | `#ffffbf` |
| 22-30° | Difficult | `#fc8d59` |
| >30° | Avoid | `#d73027` |

Permaculture guidance:
- **0-8°**: Swales, ponds, annual gardens, buildings
- **8-15°**: Food forest, perennials, light terracing
- **15-22°**: Terraces required, orchards
- **22-30°**: Minimal intervention, grazing
- **>30°**: Erosion control, timber, leave wild

### Aspect Map (Sun Exposure)
1. **Raster → Analysis → Aspect**
2. Input: smoothed DEM
3. Output is 0-360° (compass direction)

Style with 6 classes for better permaculture analysis (SE-facing is ideal in Bulgaria):

| Value | Direction | HTML Color | Notes |
|-------|-----------|------------|-------|
| 45 | Север (N) | `#5a8ac6` | Cool, shaded |
| 90 | Изток (E) | `#ffffb3` | Morning sun |
| 150 | Югоизток (SE) | `#ffd966` | Ideal - morning sun, afternoon shade |
| 210 | Юг (S) | `#f4a460` | Full sun, hot |
| 270 | Запад (W) | `#e07850` | Hot afternoon |
| 315 | Северозапад (NW) | `#c9a0dc` | Afternoon shade |
| 360 | Север (N) | `#5a8ac6` | (wrap-around) |

Use **Discrete** interpolation.

### Map Color Scheme

| Feature | HTML Color | Notes |
|---------|------------|-------|
| Buildings | `#4a4a4a` | 50% opacity |
| Major contours | `#5c4a3a` | |
| Minor contours | `#c4a882` | |
| Roads/access | `#6b8cae` | Muted blue-gray |
| Peak marker | `#e63946` | Bright red triangle |

## Output Maps

Print-ready maps created in QGIS Print Layout:

| File | Description |
|------|-------------|
| `contour_map_v1.jpg` | Topographic contour map with 1m contours |
| `slope_map_v1.jpg` | Slope analysis (5 classes for permaculture) |
| `aspect_map_v1.jpg` | Sun exposure / aspect analysis |
| `property_map_v1.jpg` | Cadastre parcels with IDs and areas over satellite |

### Map Elements
- Title in Bulgarian
- Legend with background
- Scale bar (100m)
- North arrow
- Data source and CRS attribution
- Version number

### Cadastre Labels
Expression for parcel labels with area:
```
"label" || '\n' || round("areaValue" / 1000, 1) || ' дка'
```

## Adding Bulgarian Cadastre Data

### Option 1: WMS (raster, limited styling)

1. **Layer → Add Layer → Add WMS/WMTS Layer**
2. Click **New** to create a connection
3. Enter:
   - **Name**: `BG Cadastre`
   - **URL**: `https://inspire.cadastre.bg/arcgis/services/Cadastral_Parcel/MapServer/WmsServer`
4. Click **Connect**
5. Select the cadastral parcels layer
6. **Important**: Set **Image encoding** to `PNG` and check **Transparent** to avoid white background covering satellite imagery
7. Click **Add**

### Option 2: WFS via GeoJSON (vector, full styling control)

The WFS service has a coordinate bug - it reports EPSG:4326 but returns [lat,lon] instead of [lon,lat]. Workaround:

**Step 1: Get your bounding box in lat/lon**
```python
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
lon_min, lat_min = transformer.transform(X_MIN_UTM, Y_MIN_UTM)
lon_max, lat_max = transformer.transform(X_MAX_UTM, Y_MAX_UTM)
```

**Step 2: Fetch and fix coordinates**
```bash
# Fetch GeoJSON (bbox format: min_lat,min_lon,max_lat,max_lon)
curl -s "https://inspire.cadastre.bg/arcgis/services/Cadastral_Parcel/MapServer/WFSServer?service=WFS&request=GetFeature&typeName=CP.CadastralParcel&outputFormat=geojson&bbox=41.840,24.967,41.846,24.979&count=500" -o cadastre.geojson
```

```python
import json

with open('cadastre.geojson', 'r') as f:
    data = json.load(f)

def swap_coords(coords):
    if isinstance(coords[0], list):
        return [swap_coords(c) for c in coords]
    else:
        return [coords[1], coords[0]]  # Swap lat/lon to lon/lat

for feature in data['features']:
    if feature['geometry']:
        feature['geometry']['coordinates'] = swap_coords(feature['geometry']['coordinates'])

with open('cadastre_fixed.geojson', 'w') as f:
    json.dump(data, f)
```

**Step 3: Load in QGIS**
1. **Layer → Add Layer → Add Vector Layer**
2. Select `cadastre_fixed.geojson`
3. Style as needed (Symbology → No fill, custom stroke)

### Output Files
- `cadastre_property_fixed.geojson` - 152 parcels covering the property area

**Alternative sources:**
- [INSPIRE Geoportal Bulgaria](https://inspire.egov.bg/en) - national spatial data portal
- [KAIS Portal](https://kais.cadastre.bg) - cadastre website with Open Data section
- [JOSM Maps/Bulgaria](https://josm.openstreetmap.de/wiki/Maps/Bulgaria) - lists known WMS endpoints

## Notes

- LOD 6 has ~750K points which should be sufficient for terrain/contour extraction
- For higher resolution, try LOD 5 (1.5M points) or LOD 4 (3M points) if RAM allows
- The data is from Gaussian Splatting (photogrammetry), not survey-grade LiDAR
- CSF filtering removes ~76% of points as vegetation, leaving cleaner ground surface
- For comparison with official data, EU DTM 30m dataset is available (EPSG:3035)
