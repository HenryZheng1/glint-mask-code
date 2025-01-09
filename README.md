# HARP2 Quality Flag Generator (generate_glint.py)

A Python script that processes HARP2 L1C measurement files to identify glinted angles and generate corresponding quality flags.

This tool processes HARP2 measurement files to generate quality flags (glinted angles) and creates NetCDF output files containing various atmospheric and geometric parameters.

## Overview

The quality flag generator analyzes HARP2 measurement data to identify glinted angles using both 670nm and 867nm wavelengths. It implements a smart algorithm that combines Cox-Munk model predictions with actual measurements to determine glint regions, while also accounting for cloud and land masking.

## Prerequisites

Required Python packages:
- h5py
- numpy
- matplotlib
- pandas
- xarray
- pvlib
- global_land_mask

Additional requirements:
- Water refractive index data file (path must be specified in configuration)
- HARP2 L1C measurement files (.nc format)

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install h5py numpy matplotlib pandas xarray pvlib global_land_mask
```

## Configuration

Key parameters that can be adjusted in the `main()` function:
- `angle_threshold` (default: 30.0): Maximum angle from specular reflection
- `DoLP_threshold` (default: 0.05): Maximum DoLP difference threshold
- `RI_threshold` (default: 0.3): Reflectance index threshold for cloud masking
- `WS` (default: 5.0): Wind speed for Cox-Munk model

File paths to configure:
- Input folder path for L1C files
- Output folder path for quality flag files
- Water refractive index data path (in configuration file)

## Usage

1. Update the folder paths in `main()`:
```python
folder_path = '/path/to/your/l1c/files/'
output_path = '/path/to/your/output/directory/'
```

2. Download HARP L1C files:
   - Go to the [HARP Data Portal](https://asdc.larc.nasa.gov/data/HARP2/)
   - Download L1C measurement files (.nc format)
   - Place the downloaded files in your input folder

3. Run the script:
```bash
python generate_glint.py
```

## Output

The script generates NetCDF (.nc) files containing:

### Quality Flag Data
- `qfs_670`: Quality flags/glinted angles for 670nm
- `qfs_867`: Quality flags/glinted angles for 867nm

### Geometric Parameters
- `latitude`, `longitude`: Geographic coordinates
- `VZA_670`, `VZA_867`: Viewing zenith angles
- `SZA`: Solar zenith angle
- `RAZ`: Relative azimuth angle

### Additional Fields
- `dolp_670`, `dolp_867`: Degree of Linear Polarization
- `cloud_land_mask`: Mask identifying cloud (2), land (1), and water (0) pixels

### Metadata
- File paths
- Processing parameters
- Creation timestamp

## Processing Steps

1. Reads HARP2 measurement file
2. Calculates sun-earth distance from filename
3. Retrieves and verifies viewing geometry
4. Applies Cox-Munk model for glint prediction
5. Generates cloud/land mask
6. Identifies glint regions using smart algorithm
7. Creates quality flags based on glinted angles
8. Saves results to NetCDF file

## Error Handling

The script includes comprehensive error handling:
- Angle verification at multiple stages
- Pixel-level error catching
- File processing error management

## Notes

- The quality flags (qfs) represent the number of glinted angles for each pixel
- Cloud and land pixels are automatically masked and skipped during processing
- All angles should be verified to be within expected ranges before processing
