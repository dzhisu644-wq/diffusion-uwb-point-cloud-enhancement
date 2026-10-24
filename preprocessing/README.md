# Preprocessing Boundary

The released dataset contains the processed network inputs and targets rather
than raw sensor waveforms. The conversion has three stages:

1. The NOVELDA X7 SDK performs static-background removal and coherent digital
   beamforming for 37 directions from -90 to 90 degrees.
2. The acquisition process reduces the two returned components with
   `max(abs(component))`, zeros the first 19 range bins to suppress near-field
   coupling, and applies `log1p`. Its output is a 37 x 192 log-amplitude RA
   matrix.
3. `ra_log_to_png.py` clips this matrix to [0, 3], divides by 3, transposes it,
   bilinearly resamples it to 128 x 128, and flips both axes.

Stages 2 and 3 are deterministic and are covered by the repository tests:

```bash
python preprocessing/beam_components_to_log_ra.py sdk_beams.npy frame.npy
python preprocessing/ra_log_to_png.py frame.npy frame.png
```

The paired LiDAR raster must already be represented as a 128 x 128 grayscale
PNG. Timestamp files contain one `<filename> <timestamp_ms>` pair per line.
They can be aligned with:

```bash
python preprocessing/align_pairs.py \
  --radar-dir radar_png \
  --lidar-dir lidar_png \
  --radar-timestamps radar_timestamps.txt \
  --lidar-timestamps lidar_timestamps.txt \
  --output dataset_aligned_example \
  --max-gap-ms 50
```

Raw X7 waveforms and the proprietary SDK steering coefficients are not part of
this release. Consequently, this directory reproduces dataset construction
from the SDK beamformed RA matrix onward; it does not replace the vendor SDK.
