from __future__ import print_function
from distutils import config
import os
import sys
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import List
import glob
from global_land_mask import globe
if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, base_path)
from inversion_oil import interpolate_refractive_index, read_water_refractive_index
from inversion_config import inversion_config_oil as config
from coxmunk import coxmunk
from read_HARP2 import get_HARP2_variable
import pvlib

def verify_angles(SZA, VZA, RAZ, VAA, SAA, location=""):
    """
    Verify angles are within expected ranges and print warnings if not.
    Args:
        location (str): Description of where verification is being done for better debugging
    """
    # Check for basic NaN values first
    for name, angle in [("SZA", SZA), ("VZA", VZA), ("RAZ", RAZ), 
                       ("VAA", VAA) if VAA is not None else (None, None), 
                       ("SAA", SAA) if SAA is not None else (None, None)]:
        if angle is not None and np.any(np.isnan(angle)):
            print(f"Warning: {location} - NaN values found in {name}")
    
    # Check ranges based on metadata
    if np.any((SZA < 0) | (SZA > 180)):
        print(f"Warning: {location} - SZA outside valid range [0,180]: min={np.nanmin(SZA):.2f}, max={np.nanmax(SZA):.2f}")
    
    if np.any((VZA < 0) | (VZA > 180)):
        print(f"Warning: {location} - VZA outside valid range [0,180]: min={np.nanmin(VZA):.2f}, max={np.nanmax(VZA):.2f}")
    
    if VAA is not None and np.any((VAA < -180) | (VAA > 180)):
        print(f"Warning: {location} - VAA outside valid range [-180,180]: min={np.nanmin(VAA):.2f}, max={np.nanmax(VAA):.2f}")
    
    if SAA is not None and np.any((SAA < -180) | (SAA > 180)):
        print(f"Warning: {location} - SAA outside valid range [-180,180]: min={np.nanmin(SAA):.2f}, max={np.nanmax(SAA):.2f}")
    
    # RAZ from readHARP2 is SAA-VAA without normalization
    raw_raz_min = -360
    raw_raz_max = 360
    if np.any((RAZ < raw_raz_min) | (RAZ > raw_raz_max)):
        print(f"Warning: {location} - RAZ outside expected range [{raw_raz_min},{raw_raz_max}]: min={np.nanmin(RAZ):.2f}, max={np.nanmax(RAZ):.2f}")

def get_wl(file_path, wavelength):
    """Get wavelength-specific variables from HARP2 file"""
    HARP2 = h5py.File(file_path, "r")
    
    # Calculate sun-earth distance from filename
    filename = os.path.basename(file_path)
    parts = filename.split('.')
    date_str = parts[1][:8]  # Get YYYYMMDD
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    time_index = pd.DatetimeIndex([formatted_date])
    sun_earth_dist = pvlib.solarposition.nrel_earthsun_distance(time_index, how='numpy')[0]
    print(f"Using sun-earth distance: {sun_earth_dist:.6f} AU for {formatted_date}")
    
    try:
        intensity_F0 = get_HARP2_variable(HARP2, 'intensity_F0', wavelength)[:]
    except Exception:
        print('Trying solution for non-simulated L1C data')
        intensity_F0 = get_HARP2_variable(HARP2, 'intensity_f0', wavelength)[:]
    
    # Get angles
    SZA = get_HARP2_variable(HARP2, 'solar_zenith_angle', wavelength)[:]
    VAA = get_HARP2_variable(HARP2, 'sensor_azimuth_angle', wavelength)[:]
    SAA = get_HARP2_variable(HARP2, 'solar_azimuth_angle', wavelength)[:]
    
    # Match readHARP2's RAZ calculation
    RAZ = SAA - VAA
    
    # Verify angles after initial read
    verify_angles(SZA, get_HARP2_variable(HARP2, 'sensor_zenith_angle', wavelength)[:], 
                 RAZ, VAA, SAA, "get_wl initial read")
    
    RI = get_HARP2_variable(HARP2, 'i', wavelength)[:]
    RI = ((np.pi * (sun_earth_dist**2) * RI) / (intensity_F0 * np.cos(np.deg2rad(SZA))))
    RI_stdev = get_HARP2_variable(HARP2, 'i_stdev', wavelength)[:]
    RI_stdev = np.tile(RI_stdev, (1, 1))
    DoLP = get_HARP2_variable(HARP2, 'dolp', wavelength)[:]
    VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', wavelength)[:]
    
    return SZA, VZA, RAZ, RI, DoLP

def getDoLP_CM(VZA: List[float], AZI: List[float], SZA: float, SRI: float, WS: float): 
    """Returns the Cox and Munk DoLP"""
    RM = np.zeros((4, 4, len(VZA)))
    for i in range(len(VZA)):   
        # Convert AZI to [0,180] range for Cox-Munk
        AZI_CM = np.abs(((AZI + 180) % 360) - 180)
        PHIR = np.deg2rad(AZI_CM)
        R = coxmunk(np.cos(np.deg2rad(VZA[i])), 1.0, SRI, WS, 0, 
                    np.cos(np.deg2rad(SZA)), 0, PHIR, True)
        RM[:, :, i] = np.real(R)
    DoLP_CM = np.sqrt(RM[0, 1, :]**2 + RM[0, 2, :]**2) / RM[0, 0, :]
    return DoLP_CM

def DOLP_CM_HARP2(RAZ, VZA, SZA, SRI, WS):
    """Calculate DoLP_CM for HARP2 measurements"""
    # Verify angles before Cox-Munk calculations
    verify_angles(SZA, VZA, RAZ, None, None, "Before Cox-Munk calculation")
    
    # Convert RAZ to [0,180] range for Cox-Munk
    RAZ_CM = np.abs(((RAZ + 180) % 360) - 180)
    
    DoLP_CM = []
    for i in range(len(RAZ)):
        VZA_array = np.array([VZA[i]])
        RAZ_array = np.array([RAZ_CM[i]])  # Use normalized RAZ
        SZA_array = np.array([SZA[i]])
        DoLP_CM.append(getDoLP_CM(VZA_array, RAZ_array, SZA_array, SRI, WS)[0])
    DoLP_CM = np.array(DoLP_CM)
    return DoLP_CM

def get_spec_reflec_dist(SZA, VZA, RAZ):
    """Calculate distance from specular reflection direction"""
    # Verify angles before calculation
    verify_angles(SZA, VZA, RAZ, None, None, "Before specular reflection calculation")
    
    theta_s = np.deg2rad(SZA)
    theta_v = np.deg2rad(VZA)
    phi_r = np.deg2rad(RAZ)
    
    arg = (np.cos(theta_s)*np.cos(theta_v)) - (np.sin(theta_s)*np.sin(theta_v)*np.cos(phi_r))
    return np.rad2deg(np.arccos(arg))

def get_cloud_land_mask(RI_867, RI_670, VZA_867_switch, VZA_670_switch, lat, lon, row, col, RI_threshold):
    """Generate cloud/land mask"""
    # Calculate indices
    RI_index_867 = np.argmin(np.abs(VZA_867_switch))
    RI_index_670 = np.argmin(np.abs(VZA_670_switch))
    
    # Precompute values
    RI_zero_867 = RI_867[:, :, RI_index_867]
    RI_zero_670 = RI_670[:, :, RI_index_670]
    
    # Compute land mask
    land_mask = globe.is_land(lat, lon)
    
    # Initialize mask array
    cloud_land_mask = np.zeros((row, col), dtype=np.int8)
    
    # Apply conditions
    cloud_condition = (RI_zero_867 > RI_threshold) & (RI_zero_670 > RI_threshold)
    land_condition = land_mask
    
    cloud_land_mask[cloud_condition] = 2
    cloud_land_mask[land_condition & ~cloud_condition] = 1
    
    return cloud_land_mask

def getGlintRegion_smart(DoLP, DoLP_CM, spec_reflec_dist, DoLP_threshold, angle_threshold):
    """
    Identify glint region using smart algorithm
    Returns index of first glinted angle and number of glinted angles
    """
    DoLP_Diff = np.abs(DoLP - DoLP_CM)
    valid = np.where((spec_reflec_dist < angle_threshold) & (DoLP_Diff < DoLP_threshold))
    valid = valid[0]
    if len(valid) == 0:
        return 0, 0
    index = valid[0]
    span = np.int64(valid.shape[0])
    return index, span

def generate_quality_flags(measurement_file_path, config, angle_threshold, DoLP_threshold, RI_threshold, WS):
    """
    Generate quality flags for a HARP2 measurement file
    
    Args:
        measurement_file_path: Path to HARP2 measurement file
        config: Configuration object containing paths and parameters
        angle_threshold: Maximum angle from specular reflection
        DoLP_threshold: Maximum DoLP difference threshold
        RI_threshold: Reflectance index threshold for cloud masking
        WS: Wind speed for Cox-Munk model
    
    Returns:
        Dictionary containing quality flags and associated data
    """
    # Open HARP2 file
    HARP2 = h5py.File(measurement_file_path, "r")
    
    bins_along_track, bins_across_track, number_of_views, intensity_bands = HARP2['observation_data/i'].shape
    
    # Get basic dimensions
    row, col = bins_along_track, bins_across_track
    
    # Get refractive indices
    water_refindx_data = read_water_refractive_index(config.water_rfindx_path)
    SRI_670, _ = interpolate_refractive_index(670., water_refindx_data)
    SRI_867, _ = interpolate_refractive_index(867., water_refindx_data)
    
    # Get viewing angles
    VZA_670_switch = get_HARP2_variable(HARP2, 'sensor_view_angle', 670)
    VZA_867_switch = get_HARP2_variable(HARP2, 'sensor_view_angle', 867)
    
    # Get wavelength-specific data
    SZA_670, VZA_670, RAZ_670, RI_670, DoLP_670 = get_wl(measurement_file_path, 670)
    SZA_867, VZA_867, RAZ_867, RI_867, DoLP_867 = get_wl(measurement_file_path, 867)
    
    # Get lat/lon for cloud masking
    lat = get_HARP2_variable(HARP2, 'latitude', 867)
    lon = get_HARP2_variable(HARP2, 'longitude', 867)
    
    # Generate cloud/land mask
    cloud_land_mask = get_cloud_land_mask(RI_867, RI_670, VZA_867_switch, VZA_670_switch, 
                                        lat, lon, row, col, RI_threshold)
    
    # Initialize quality flag arrays
    qfs_670 = np.zeros(shape=(row, col), dtype=np.int8)
    qfs_867 = np.zeros(shape=(row, col), dtype=np.int8)
    
    # Process each pixel
    for r in range(row):
        for c in range(col):
            # Skip invalid pixels
            if np.isnan(VZA_867[r, c]).all() or np.isnan(RI_867[r, c]).all():
                continue
            if cloud_land_mask[r][c] >= 1:
                continue
                
            try:
                # Calculate Cox-Munk predictions
                DOLP_CM_670 = DOLP_CM_HARP2(RAZ_670[r, c], VZA_670[r, c], SZA_670[r, c], SRI_670, WS)
                DOLP_CM_867 = DOLP_CM_HARP2(RAZ_867[r, c], VZA_867[r, c], SZA_867[r, c], SRI_867, WS)
                
                # Calculate specular reflection distances
                spec_reflec_dist_670 = get_spec_reflec_dist(SZA_670[r, c], VZA_670[r, c], RAZ_670[r, c])
                spec_reflec_dist_867 = get_spec_reflec_dist(SZA_867[r, c], VZA_867[r, c], RAZ_867[r, c])
                
                # Get glint regions
                _, span_670 = getGlintRegion_smart(DoLP_670[r, c], DOLP_CM_670, spec_reflec_dist_670,
                                                 DoLP_threshold=DoLP_threshold, angle_threshold=angle_threshold)
                _, span_867 = getGlintRegion_smart(DoLP_867[r, c], DOLP_CM_867, spec_reflec_dist_867,
                                                 DoLP_threshold=DoLP_threshold, angle_threshold=angle_threshold)
                
                qfs_670[r, c] = span_670
                qfs_867[r, c] = span_867
                
            except (IndexError, ValueError) as err:
                print(f"Error processing pixel ({r}, {c}): {err}")
                continue
    
    # Create output dataset
    measurement_filename = os.path.basename(measurement_file_path)
    out = xr.Dataset({
        'qfs_670': (['row', 'col'], qfs_670),
        'qfs_867': (['row', 'col'], qfs_867),
        'latitude': (['row', 'col'], lat),
        'longitude': (['row', 'col'], lon),
        'VZA_670': (['numberofviews_670'], VZA_670_switch),
        'VZA_867': (['numberofviews_867'], VZA_867_switch),
        'SZA': (['row', 'col', 'number_of_views'], SZA_867),
        'RAZ': (['row', 'col', 'number_of_views'], RAZ_867),
        'dolp_670': (['row', 'col', 'numberofviews_670'], DoLP_670),
        'dolp_867': (['row', 'col', 'numberofviews_867'], DoLP_867),
        'cloud_land_mask': (['row', 'col'], cloud_land_mask)
    },
    coords={
        'row': np.arange(row), 
        'col': np.arange(col),
        'numberofviews_670': np.arange(len(VZA_670_switch)),
        'numberofviews_867': np.arange(len(VZA_867_switch)),
        'number_of_views': np.arange(SZA_867.shape[-1])
    },
    attrs={
        'measurement_file_path': measurement_file_path,
        'measurement_filename': measurement_filename,
        'angle_threshold': angle_threshold,
        'DoLP_threshold': DoLP_threshold,
        'RI_threshold': RI_threshold,
        'wind_speed': WS,
        'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return out

def main():
    """Main execution function"""
    # Configuration
    angle_threshold = 30.0  # Maximum angle from specular reflection
    DoLP_threshold = 0.05  # Maximum DoLP difference threshold
    RI_threshold = 0.3    # Reflectance index threshold for cloud masking
    WS = 5.0            # Wind speed for Cox-Munk model
    
    # Process files
    folder_path = '/Users/henry/Documents/temp_l1c/'  # Update with your folder path
    output_path = '/Users/henry/Documents/temp_l1c/QF_output'   # Update with your output path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process all nc files in folder
    for file_path in glob.glob(os.path.join(folder_path, '*.nc')):
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            
            # Generate quality flags
            result = generate_quality_flags(file_path, config, 
                                         angle_threshold, DoLP_threshold, 
                                         RI_threshold, WS)
            
            # Save results
            output_file = os.path.join(output_path, 
                                     f"QF_{os.path.basename(file_path)}")
            result.to_netcdf(output_file)
            print(f"Saved quality flags to {output_file}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()