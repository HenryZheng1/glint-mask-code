# HARP2 Quality Flag Generator (generate_glint.py)

A Python script that processes HARP2 L1C measurement files to identify “glinted” angles for each pixel and produce a single numeric indicator called a *quality flag*. Specifically:

- **At 670 nm**, HARP2 collects **60 angles**.
- **At 867 nm**, HARP2 collects **10 angles**.

For each pixel in the image, the *quality flag* is the **count of how many** of those angles meet the “glint” criteria (e.g., 0 = no glint, 1 = one angle glinted, etc.).

This tool also creates NetCDF output files containing various atmospheric and geometric parameters (e.g., latitude, longitude, masks).

---

## Overview

This script uses a combination of the Cox–Munk model (a classical model for specular reflection off water surfaces) and the HARP2 measurements to detect glint. It also accounts for cloud and land masking, so those pixels are excluded from glint detection. The main steps include:

1. Reading each HARP2 L1C file  
2. Determining geometric parameters (viewing angles, sun–earth distance, etc.)  
3. Predicting specular reflection via Cox–Munk  
4. Applying cloud‐and‐land masks  
5. Checking each of the 60 angles at 670 nm and each of the 10 angles at 867 nm for glint  
6. Generating one “quality flag” value per pixel (the number of glinted angles)  
7. Saving results to a NetCDF file  

---

## Prerequisites

Required Python packages:

- `h5py`  
- `numpy`  
- `matplotlib`  
- `pandas`  
- `xarray`  
- `pvlib`  
- `global_land_mask`

Additional requirements:

- **Water refractive index data file** (path must be specified in configuration)  
- **HARP2 L1C measurement files** in `.nc` format  

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/harp2-glint-flags.git
   ```
2. **Install required packages**:
   ```bash
   pip install h5py numpy matplotlib pandas xarray pvlib global_land_mask
   ```

---

## Configuration

In the `main()` function, you can configure:

- `angle_threshold` (default: 30.0) – Maximum angle from specular reflection  
- `DoLP_threshold` (default: 0.05) – Threshold for degree of linear polarization  
- `RI_threshold` (default: 0.3) – Reflectance index threshold for cloud masking  
- `WS` (default: 5.0) – Wind speed used in the Cox–Munk model  

File paths to set (also in `main()` or a config file):

- **Input folder** for L1C files  
- **Output folder** for NetCDF results  
- **Water refractive index data** file path  

---

## Usage

1. **Edit the folder paths** in `main()` (or your config file):
   ```python
   folder_path = '/path/to/your/l1c/files/'
   output_path = '/path/to/your/output/directory/'
   ```
2. **Download HARP2 L1C files**  
   - Visit the [HARP Data Portal](https://asdc.larc.nasa.gov/data/HARP2/)  
   - Download `.nc` L1C measurement files  
   - Place them in the input folder
3. **Run the script**:
   ```bash
   python generate_glint.py
   ```

---

## Output

Running the script produces NetCDF (`.nc`) files with:

- **`qfs_670`** – Number of “glinted” angles at 670 nm (out of 60 possible angles)  
- **`qfs_867`** – Number of “glinted” angles at 867 nm (out of 10 possible angles)  

In addition, each file contains:

- **Geometric parameters**:  
  - `latitude`, `longitude`, `VZA_670`, `VZA_867`, `SZA`, `RAZ`  
- **Additional fields**:  
  - `dolp_670`, `dolp_867`: Degree of linear polarization  
  - `cloud_land_mask`: Cloud (2), land (1), water (0)  
- **Metadata**: file paths, processing parameters, timestamps  

---

## Processing Steps

1. **Read** each HARP2 measurement file  
2. **Compute** sun–earth distance from filename/time metadata  
3. **Extract & verify** viewing geometry (zenith/azimuth angles)  
4. **Apply** Cox–Munk model to estimate expected glint conditions  
5. **Build** a cloud & land mask (using reflectance and land boundary data)  
6. **Identify** glint in each of the 60 angles at 670 nm and 10 angles at 867 nm  
7. **Form** the *quality flag* per pixel as the count of glinted angles  
8. **Write** the output to a NetCDF file  

---

## Error Handling

- Checks that angles, geometry, and file inputs are valid  
- Catches pixel‐level anomalies (e.g., missing data)  
- Logs file I/O issues and continues with remaining files  

---

## Notes

- Cloud or land pixels automatically skip glint detection  
- All viewing angles and sensor geometry should be validated before processing  
- The “quality flag” simply indicates how many of the measured angles at a given wavelength are glinted
