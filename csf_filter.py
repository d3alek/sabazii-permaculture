#!/usr/bin/env python3
"""Filter ground points from photogrammetry point cloud using CSF."""

import numpy as np
import CSF

# Load points
print("Loading points...")
data = np.loadtxt('georeferenced_xyz.csv', delimiter=',', skiprows=1)
print(f"Loaded {len(data):,} points")

# Configure CSF
csf = CSF.CSF()
csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 1.0  # meters, lower = more detail
csf.params.rigidness = 2           # 1=flat, 2=gentle slope, 3=steep
csf.params.time_step = 0.65
csf.params.class_threshold = 0.5   # lower = stricter ground classification
csf.params.interations = 500

# Run filter
print("Running cloth simulation filter...")
csf.setPointCloud(data)
ground_idx = CSF.VecInt()
offground_idx = CSF.VecInt()
csf.do_filtering(ground_idx, offground_idx)

# Extract results
ground_points = data[list(ground_idx)]
offground_points = data[list(offground_idx)]

print(f"Ground points: {len(ground_points):,}")
print(f"Off-ground (vegetation): {len(offground_points):,}")
print(f"Ratio: {len(ground_points)/len(data)*100:.1f}% classified as ground")

# Save ground points
np.savetxt('ground_points.csv', ground_points, delimiter=',',
           header='x,y,z', comments='', fmt='%.6f')
print("Saved ground_points.csv")
