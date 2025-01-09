#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:10:09 2022

@author: cyberbass
"""
from lmfit import Parameters
import os
import numpy as np

class inversion_config_oil:
    folder_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    code_dir = os.path.dirname(folder_dir)
    rt_root_dir = code_dir + "/rt_code/"
    rt_dir = rt_root_dir + 'rt_code'
    water_rfindx_path = folder_dir + '/tables/water/water_rfindx_segelstein.txt'
    src_dir = os.path.dirname(__file__)
    data_dir = code_dir + "/data/"
    merra2_data_dir = data_dir + "/merra2_data/"
    merra2_hdf_dir = data_dir + "/merra2_hdf/"
    ###########################################
    #include both x1 and x2 transect flights right here figure out how to do this later
    x1_transect = ""
    x2_transect = ""
    
    switch_I                 = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    switch_polarization      = [0, 0, 0, 0, 0, 0, 0, 0, 1] # 0 = not use, 1 = use DoLP, 2 = use polarized reflectance
    switch_I = [1]
    switch_polarization = [1]
    switch_oil_detection     = 1 # 0 = do not use, 1 = use glint region for oil detection #! this is not fully implemented 
    '''
    dataType:
    0 = synthetic                Files can be created by using the create_synthetic.py file.
    1 = RSP new format (L1C)     Files can be found at: https://data.giss.nasa.gov/pub/rsp/data/ under the campaign name and the L1C folder.
    2 = RSP old format (L1B)     Files can be found at: https://data.giss.nasa.gov/pub/rsp/data-old-format/ under the campaign name and the L1B folder.
    3 = HARP-2                   Simulated data can be found at: https://oceandata.sci.gsfc.nasa.gov/directdataaccess/Level-1C/PACE_HARP2/.
    4 = POLDER-MODIS             h5 files subsampled to Greenland.
    5 = eGAP                     netcdf files produced with Jacek's RTC (microplastics).
    '''
    dataType = 3
    layer_m = [0.0, 2000.0, 5000.0, 10000.0, 100000.0] # layers of atmosphere in meters. If using dataType = 1/2 the alt. of the plane will be added to the array later
    #name of file to be read in
    #measurement_file_name = 'RSP2-ER2_L1B-RSPGEOL1B-GeolocatedRadiances_20130916T154659Z_V003-20170216T221924Z.h5'    # L1B S2
    #measurement_file_name = 'RSP1-B200_L1B-RSPGEOL1B-GeolocatedRadiances_20100511T191536Z_V001-20150414T041430Z.h5'   # L1B X2
    #measurement_file_name = 'RSP1-B200_L1B-RSPGEOL1B-GeolocatedRadiances_20100511T183620Z_V001-20150414T042120Z.h5'   # L1B X1
    measurement_file_name = '/RSP/RSP2-ER2_L1C-RSPCOL-CollocatedRadiances_20130916T154659Z_V006-20230802T043145Z.h5'  # L1C S2
    # measurement_file_name = '/RSP/RSP1-B200_L1C-RSPCOL-CollocatedRadiances_20100511T191613Z_V003-20231027T191108Z.h5'   # L1C X2
    # measurement_file_name = '/RSP/RSP1-B200_L1C-RSPCOL-CollocatedRadiances_20100511T183802Z_V003-20231027T190417Z.h5'   # L1C X1
    measurement_file_name = '/PACE/PACE_HARP2_SIM.20220321T183042.L1C.5km.V03.nc'
    # measurement_file_name = '/PACE/PACE_HARP2_SIM.20220321T183042.L1C.5km.V03.nc'


    measurement_file_path = data_dir + measurement_file_name
    merra2_hdf_path = merra2_hdf_dir + 'MERRA2_' + measurement_file_name[:-2] + 'hdf'
    HARP2_ancillary = data_dir + '/PACE/PACE.20220321T183042.L1C.5.2km.ANC.nc' 


    #start and end scan to be used in inversion (can be found in 2019 paper)
    #start_scan, end_scan = 2800, 2819
    start_scan, end_scan = 100, 1800 # L1C X2
    #start_scan, end_scan = 100, 1000 # L1C X1
    row_pixel, col_pixel = 95, 339
    
    #number of scans to be used
    nscans = 20
    
    #output_file_name is what retrieval outputs will be saved as
    output_file_name = "X2"
    
    # solar zenith angle for eGAP files
    sza_eGAP = 10.
    
    #use standard deviation during inversion
    use_stdev = False

    # correct for gas absorption during inversion
    correct_gas_abs = True
    xo3col, xno2col, xh2ocol = 0.3, 0, 5.00318
    
    #inversion settings
    epsfcn=0.01 # epsfcn controls the step size in Jacobian calculation, typical range between 0.05 (large) - 0.0001 (small)
    ftol=1.e-2   # ftol is the tolerance threshold for residual
    xtol=1.e-1
    gtol=1.e-8
    max_nfev=300   # max_nfev is the maximum number of function evaluation
    
    # epsfcn=0.01 
    # ftol=1.e-22   
    # xtol=1.e-10   
    # gtol=1.e-22  
    # max_nfev=300
    
    freeParams = Parameters()
    # Add parameters here and implement the relevant logic in inversion_oil.py to replace default values defined in atm_surf_pars.py
    freeParams.add("Windspeed", value = 5.0,      min = 1.0,  max = 15.0)  # windspeed
    freeParams.add("NRsrf",     value = 1.28,     min = 1.1, max = 2.0)  # real surface refractive index
    freeParams.add("NZc",       value = 0.02,    min = 0.0,  max = 0.3)   # coarse mode aerosol optical thickness
    freeParams.add("NZf",       value = 0.001,    min = 0.0,  max = 0.1, vary=False)   # fine mode aerosol optical thickness

## For the snow inversion config class    
class inversion_config_snow: 
    
    code_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # code_dir = '/Users/olivia/Downloads/code/snow_code/'
    rt_root_dir = code_dir + "/rt_code/"
    # rt_root_dir = '/Users/olivia/Downloads/snow_RT/' # This line is because of the line length restriction of setupatm.f
    rt_dir = rt_root_dir + 'rt_code'
    scatt_root_dir = code_dir + '/data/ice_Scat_Matrices/'
    water_rfindx_path = code_dir + '/tables/water/water_rfindx_segelstein.txt'
    src_dir = os.path.dirname(__file__)
    data_dir = code_dir + "/data/"    
   
    ## synthetic data
    # wavelength_list_mono = np.array([554.93747, 469.2736, 670.02308, 863.51, 1240, 1629.282, 2111.9325]) # wavelength list for mono angle instruments
    # wavelength_list = np.array([554.93747, 469.2736, 670.02308, 863.51, 1588.5468, 2111.9325, 2265.5691]) # wavelength list for polarized multi-angle instruments
    # ref_wl = 554.93747 # reference wavelength for optical depth calculation  

    # switch_I_mono = [1, 1, 1, 1, 0, 1, 1]    
    
    # switch_I = [1, 0, 0, 0, 0, 0, 0]
    # switch_DoLP = [0, 0, 0, 0, 0, 0, 0]
    # switch_Rp = [0, 0, 0, 0, 0, 0, 0]  
    
    #### PM data
    wavelength_list_mono = np.array([465.692282, 553.688049, 645.813704, 856.115699, 1242.03146, 1629.28772, 2112.98490]) # wavelength list for MODIS #! not ice mat for 1240
    wavelength_list = np.array([490., 670.,  865.]) # wavelength list for POLDER
      
    ref_wl = 465.692282
    switch_I_mono = [1, 1, 1, 1, 0, 1, 1] 
    switch_I =    [1, 1, 1]
    switch_DoLP = [0, 0, 0]
    switch_Rp =   [1, 1, 1]
    
    I_VZA_angles = None
    DoLP_VZA_angles = None
    Rp_VZA_angles = None

    #inversion settings
    epsfcn=0.0001 # epsfcn controls the step size in Jacobian calculation, typical range between 0.05 (large) - 0.0001 (small)
                # A variable used in determining a suitable step length for the forward- difference approximation of the Jacobian (for Dfun=None). 
                # Normally the actual step length will be sqrt(epsfcn)*x 
                # If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision
    ftol=1.e-8   # ftol is the relative error desired in the sum of squares
    xtol=1.e-8   # xtol is the relative error desired in the approximate solution
    gtol=1.e-8    # gtol is the orthogonality desired between the function vector and the columns of the Jacobian
    max_nfev=300   # max_nfev is the maximum number of function evaluation

    '''
    dataType: 
    0 = synthetic            Files can be created by using the create_synthetic.py file.
    1 = POLDER-MODIS         h5 files subsampled to Greenland.
    '''
    dataType = 1
    PM_file_path = "/Users/olivia/PM/PM01_L2.GRL_2008_07_07T14-33-44.h5"#data_dir + "PM01_L2.GRL_2008_07_07T14-33-44.h5"
    MERRA2_file_path = data_dir + "MERRA2_300.inst1_2d_asm_Nx.20080707.SUB.nc"
    row_pixel, col_pixel = 300, 150
    SCM_file_path = data_dir + 'CALTRACK-333m_SCM_V1-1_2007-08-09T14-56-43ZD.hdf'
    
    nr_imp = 1.80 # real refractive index of snow impurity
    ni_imp = 0.6 # imaginary refractive index of snow impurity
    rho_imp = 2.0 # density of snow impurity[g/cm^3]
    mu = 0.13100  # aerosol mu
    sigma = 0.380 # aerosol sigma

    freeParams = Parameters()
    freeParams.add("aod",       value = 0.05,    min = 0.0,  max = 1.0,   vary = 0) # aerosol optical thickness 
    freeParams.add("d_1",       value = 0.5,     min = 0.05,   max = 0.7,     vary = 0) # roughness top layer (0=pristine, 1=rough)
    freeParams.add("d_2",       value = 0.4,     min = 0.05,   max = 0.7,     vary = 0) # roughness bottom layer
    freeParams.add("ar_1",      value = 0.05,     min = 0.038,  max = 26.7, vary = 0) # aspect ratio top layer (snow grain shape) 
    freeParams.add("ar_2",      value = 0.08,      min = 0.038,  max = 26.7, vary = 0) # aspect ratio bottom layer (0=plates, 27=columns)
    freeParams.add("reff_1",     value = 250.0,     min = 56.0,    max = 1560.0,  vary = 0) # effective radius top layer (snow grain size) [micron]
    freeParams.add("reff_2",     value = 500.0,    min = 56.0,    max = 1560.0,  vary = 0) # effective radius bottom layer (snow grain size) [micron]
    freeParams.add("f_1",       value = 0.5,     min = 0.0,    max = 1,        vary = 0) # fraction of plates to columns top layer
    freeParams.add("f_2",       value = 0.5,     min = 0.0,    max = 1,       vary = 0) # fraction of plates to columns  bottom layer
    freeParams.add("soot_1",    value = 0.0,     min = 0.0001,   max = 1.,    vary = 0) # soot concentration ratio top layer [ppmw]
    # soot_2 is fixed to be equal to soot_1 in calculate_residual for the current version (Oct.25, 2023)
    # freeParams.add("soot_2",    value = 0.01,     min = 0.0001,   max = 1.,    vary = 0) # soot concentration bottom layer [ppmw]
    freeParams.add("dens_1",     value = 0.2,      min = 0.0,    max = 0.917,        vary = 0) # snow density of the top layer [g/cm^3]
    freeParams.add("dens_2",     value = 0.3,      min = 0.0,    max = 0.917,   vary = 0) # snow density of the bottom layer [g/cm^3]
    freeParams.add("thick_1",     value = 0.1,      min = 0.005,    max = 1.0,   vary = 0) # thickness of the top layer snow [meter]    
    
    #################################################################
    ####    Setting the free parameters you want to vary to 1    ####
    #################################################################
    freeParams['aod'].vary = 1
    freeParams['reff_1'].vary = 1
    freeParams['reff_2'].vary = 1
    # freeParams['d_1'].vary = 1
    # freeParams['ar_1'].vary = 1        
    # freeParams['soot_1'].vary = 1
    
    