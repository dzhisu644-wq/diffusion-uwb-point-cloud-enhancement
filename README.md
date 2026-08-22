# Diffusion-Based Real-Time Point-Cloud Enhancement for UWB Radar

This repository accompanies the work **Diffusion-Based Real-Time Point-Cloud
Enhancement for UWB Radar**. It provides the processed, synchronized UWB radar
and LiDAR dataset used for point-cloud enhancement and generalization
evaluation.

<p align="center">
  <img src="assets/experiment_platform.png" width="620"
       alt="Mobile sensing platform with the NOVELDA X7 UWB radar and RPLIDAR A1">
</p>

## Hardware

- **Input sensor:** NOVELDA X7 impulse-UWB radar.
- **Reference sensor:** SLAMTEC RPLIDAR A1 2-D LiDAR, used to provide synchronized
  geometric targets.

## Dataset

The dataset contains 113,204 synchronized radar-LiDAR pairs collected with a
NOVELDA X7 UWB radar and an RPLIDAR A1 on a mobile platform. Twelve indoor
scenes (93,518 pairs) form the training domain. The physically separate L2
scene (19,686 pairs) is reserved for final generalization evaluation.

Each sample consists of two 128 x 128, 8-bit grayscale PNG images on a common
range--azimuth grid:

```text
X7_A1_dataset/
  dataset_aligned_<scene>/
    radar/000000.png
    lidar/000000.png
```

Files with the same name form one synchronized pair. The released scenes are
`D7`, `G7`, `H`, `J2`, `K1`, `K5`, `L10`, `L1_S`, `L6`, `T3`, `T5`, `T6`, and
`L2`.

### Example Synchronized Pair

<table>
  <tr>
    <th>UWB range-azimuth input</th>
    <th>LiDAR target</th>
  </tr>
  <tr>
    <td><img src="assets/example_uwb_ra.png" width="256" alt="UWB range-azimuth input"></td>
    <td><img src="assets/example_lidar_target.png" width="256" alt="Synchronized LiDAR target"></td>
  </tr>
</table>

The example above is the synchronized pair `dataset_aligned_D7/000499.png`.

## Download

The complete dataset is distributed as a GitHub Release asset:

```bash
wget https://github.com/dzhisu644-wq/diffusion-uwb-point-cloud-enhancement/releases/download/dataset-v1.0/X7_A1_dataset_v1.tar.zst
wget https://github.com/dzhisu644-wq/diffusion-uwb-point-cloud-enhancement/releases/download/dataset-v1.0/X7_A1_dataset_v1.tar.zst.sha256
sha256sum -c X7_A1_dataset_v1.tar.zst.sha256
tar -I zstd -xf X7_A1_dataset_v1.tar.zst
```

Training and evaluation code will be added to this repository. Citation
information will be provided upon publication.
