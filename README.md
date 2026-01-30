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

## QGIS Workflow

### Step 1: Import point cloud
1. **Layer → Add Layer → Add Delimited Text Layer**
2. File: `georeferenced_xyz.csv`
3. X field: `x`, Y field: `y`, Z field: `z`
4. Geometry CRS: `EPSG:32635`

### Step 2: Create DEM (TIN interpolation)
1. **Processing → Toolbox** → search "TIN interpolation"
2. Input layer: georeferenced_xyz
3. Interpolation attribute: `z`
4. Pixel size: `1` (1 meter)
5. Run

### Step 3: Smooth the DEM
The raw DEM has noise from vegetation and surface detail. Smooth it before contouring:

1. **Processing Toolbox** → search "Warp"
2. Select **GDAL → Warp (reproject)**
3. Input: TIN interpolation raster
4. Output resolution: `5` (5 meters)
5. Resampling method: `average`
6. Run

*Alternative: Use r.neighbors (GRASS) with neighborhood size 11 if available.*

### Step 4: Generate contours
1. **Raster → Extraction → Contour**
2. Input layer: the smoothed/warped raster
3. Interval: `1` (1 meter contours)
4. Run

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
- **Vegetation issue**: Photogrammetry captures canopy surface, not ground. The smoothing step helps reduce this noise, but forested areas will still show canopy elevation. For accurate ground contours in vegetated areas, consider:
  - Cloth Simulation Filter (CSF) in CloudCompare
  - Focus on open/cleared areas only
  - Use minimum Z value per grid cell instead of TIN
