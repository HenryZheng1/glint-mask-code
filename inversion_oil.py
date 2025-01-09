#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 02:46:53 2022

@author: cyberbass
"""

import os, time, shutil, glob, subprocess, sys
from multiprocessing import Queue, Process, cpu_count
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import xarray as xr
import fortranformat as ff
from fortranformat import FortranRecordWriter as ffFRW
from lmfit import minimize, Minimizer
from scipy.interpolate import interp1d
# from collections import defaultdict
from lib.coxmunk import getDoLP_CM, getGlintRegion
from lib.create_rt_input_files import Atmo, Srf
from lib.altitude_to_pressure import altToPressure
from pyhdf.SD import *
from lib.absorption_profile import absorption_profile
# from inversion_config_oil import *
from plot.plot_stokes import Inversion_GIF, create_GIF, inv_iteration_plot, create_inversion_netcdf
from reader.read_model import readModel

def read_ice_refractive_index(file_path):

    data = pd.read_csv(file_path, header=None, skiprows=7, delimiter=r"\s+", names=['Wavelength','n_r','n_i']) 

    return data

def read_water_refractive_index(file_path):

    data = pd.read_csv(file_path, header=None, skiprows=4, delimiter=r"\s+", names=['Wavelength','n_r','n_i'])

    return data

def interpolate_refractive_index(wavelengths, df_rfindx):
    # input wavelengths should be in nm
    # The wavelength in water refractive index table (Segelstein 1981) is in micron
    wavelengths_um = wavelengths/1000.

    f_nr = interp1d(df_rfindx.Wavelength, df_rfindx.n_r)
    n_r_int = f_nr(wavelengths_um)
    f_ni = interp1d(df_rfindx.Wavelength, df_rfindx.n_i)
    n_i_int = f_ni(wavelengths_um)
    # print(n_r_int)
    return n_r_int, n_i_int

def read_inversion_input_file(measurement_file_path, dataType, start_scan=0, end_scan=20, wl=867., row=0, col=0):
    from inversion_config import inversion_config_oil as config
    if dataType == 1: # RSP new format (L1C)
        from reader.read_RSP import readRSP
        measurement_data = readRSP(measurement_file_path, start_scan, end_scan)
    if dataType == 2: # RSP old format (L1B)
        from reader.read_RSP_oldformat import readRSP_oldformat
        measurement_data = readRSP_oldformat(measurement_file_path, start_scan, end_scan)
    if dataType == 3:
         from reader.read_HARP2 import readHARP2
         measurement_data = readHARP2(measurement_file_path, wl, row, col)
    if dataType == 4:
        from reader.read_POLDER import readPOLDER
        measurement_data = readPOLDER(measurement_file_path, wl, row, col)
    if dataType == 5:
        from reader.read_eGAP import read_eGAP
        measurement_data = read_eGAP(measurement_file_path, config.sza_eGAP)
    # if dataType == 6:
        # measurement_data = readRSP_PACE(measurement_file_path)

    return measurement_data

def get_glint_index(data, SRI, WS, HARP2=False):

    if data.Wavelength.size > 1:
        raise TypeError(f"The data is not properly sliced. Number of wavelengths is {data.Wavelength.size} (should be 1)")

    ## Calculate the Cox-Munk DoLP
    if not HARP2:
        VZA_array = data.VZA.values
        RAZ_array = data.mean_RAZ.values
        SZA = data.mean_SZA.values
        DoLP_CM = getDoLP_CM(VZA_array, RAZ_array, SZA, SRI, WS)
        # specularidx = np.where(data.RI.values == np.nanmax(data.RI.values))[0] # index of maximum total reflectance
        specularidx = np.where(data.RI.values==np.nanmax(data.RI.values[np.abs(data.SZA.values-data.VZA.values)<15.]))[0][0]
        index, span = getGlintRegion(data.DoLP.values, DoLP_CM, data.VZA.values, specularidx)

    else:
        DoLP_CM = []
        for i in range(len(data.RAZ)):
            VZA_array = np.array([data.VZA.values[i]])
            RAZ_array = np.array([data.RAZ.values[i]-180.]) #!!!!!
            SZA = np.array([data.SZA.values[i]])
            DoLP_CM.append(getDoLP_CM(VZA_array, RAZ_array, SZA, SRI, WS)[0])
        DoLP_CM = np.array(DoLP_CM)
        ## Get glint index and span
        specularidx = np.where(data.RI.values==np.nanmax(data.RI.values[np.abs(data.SZA.values-data.VZA.values)<15.]))[0][0] # index of maximum total reflectance
        if data.Wavelength.values == 670.:
            index, span = getGlintRegion(data.DoLP.values, DoLP_CM, data.VZA.values, specularidx, nangles_min=16)
        else:
            index, span = getGlintRegion(data.DoLP.values, DoLP_CM, data.VZA.values, specularidx, nangles_min=4)
    
    return index, span

def get_active_wavelengths(arr_wavelengths, switch_I, switch_polarization):
    arr_switch_I = np.array(switch_I, )
    arr_switch_polarization = np.array(switch_polarization)
    arr_switch_pol = arr_switch_polarization
    arr_switch_pol[arr_switch_polarization>0]=1
    wavelength_switch = np.bitwise_or(arr_switch_I, arr_switch_pol)
    wavelengths_active = arr_wavelengths[wavelength_switch==1]

    return wavelengths_active

def data_stitcher(data, index, span, wavelengths_active, stdev=False):
    from inversion_config import inversion_config_oil as config
    switch_oil_detection = config.switch_oil_detection

    mono_array = []
    if not stdev:
        for wl in wavelengths_active:

            # add if switches here later
            RI = data.RI.sel(Wavelength=wl, method='nearest')
            mono_array.append(RI.values)

        # add Rp switches here later
        # Rp = data.Rp.sel(Wavelength=wl, method='nearest')

        for wl in wavelengths_active:
            # add if switches here later
            DoLP = data.DoLP.sel(Wavelength=wl, method='nearest')
            if switch_oil_detection == 1:
                mono_array.append(DoLP.values[index:span])
            else:
                mono_array.append(DoLP.values)

    else:
        for wl in wavelengths_active:

            # add if switches here later
            RI = data.RI_stdev.sel(Wavelength=wl, method='nearest')
            mono_array.append(RI.values)

        # add Rp switches here later
        # Rp = data.Rp.sel(Wavelength=wl, method='nearest')

        for wl in wavelengths_active:
            # add if switches here later
            DoLP = data.DoLP_stdev.sel(Wavelength=wl, method='nearest')
            if switch_oil_detection == 1:
                mono_array.append(DoLP.values[index:span])
            else:
                mono_array.append(DoLP.values)

    mono_array = np.hstack(mono_array)

    return mono_array

def plot_inversion_diagnostics(measurement_data, model_data, dataType, wavelength, freeParams, text_file_name, plot_dir, scattering_angle=False):
    # find RI, Rp, and DoLP limits based on maximum values
    data = measurement_data.sel(Wavelength=wavelength, method='nearest')
    RI_lim = np.nanmax(data.RI)+np.nanmax(data.RI)*0.1 if 'RI' in data.variables else 1.0
    Rp_lim = np.nanmax(data.Rp)+np.nanmax(data.Rp)*0.1 if 'RI' in data.variables else 1.0
    DoLP_lim = np.nanmax(data.DoLP)+np.nanmax(data.DoLP)*0.1 if 'RI' in data.variables else 1.0

    text = [f'{key}: {val.value:.2f}' for key, val in freeParams.items() if val.vary==1] # create list of parameter and values of the parameter
    Inversion_GIF(plot_dir, measurement_data, model_data, dataType, [wavelength], RI_lim, Rp_lim, DoLP_lim, text, scattering_angle=scattering_angle) # call the plotting routine
    # create file that holds all of the values of the paramters being values
    working_dir = os.path.dirname(os.path.dirname(plot_dir))
    with open(f'{working_dir}/{text_file_name}.txt', "a+") as f:
        f.seek(0)             # Move read cursor to the start of file.
        data = f.read(100) # If file is not empty then append '\n'
        if len(data) > 0 :
            f.write("\n")
        param_values = [f'{val.value:.5f}' for key, val in freeParams.items() if val.vary==1]
        for item in param_values:
            f.write(f'{item} ') # keep extra space in the string
        f.close() #close the file
    return  

def calculate_residual_oil(freeParams, measurement_data, index, span, wavelength, DELP, TAUABS, water_refindx_data, use_stdev, working_dir, text_file_name, plot_dir):
    from inversion_config import inversion_config_oil as config
    switch_oil_detection = config.switch_oil_detection
    rt_dir = config.rt_dir
    dataType = config.dataType
    from atm_surf_params import atm_surf_pars_oil

    atmosphere = atm_surf_pars_oil.atmosphere
    surface = atm_surf_pars_oil.surface

    print('Free parameter values: ')
    for key, val in freeParams.items():
        if val.vary == 1:
            print(f'{key:<10s}: {val.value:.5f}')

    ## Stitch the RSP data into a 1-D array
    measurement_1d_array = data_stitcher(measurement_data, index, span, [wavelength])
    ## run RT code to compute model data
    n_r_surf, n_i_surf = interpolate_refractive_index(wavelength, water_refindx_data)

    wavelength_um = wavelength/1000.
    atmosphere['ALAM'] = wavelength_um
    atmosphere['DELP'] = DELP
    atmosphere['TAUABS'] = TAUABS
    surface['F5'] = [n_r_surf]
    surface['F6'] = [n_i_surf]

    if switch_oil_detection == 1:
        rsp_muSZA = np.cos(np.deg2rad(np.nanmean(measurement_data.SZA.values[index:span])))
    else:
        rsp_muSZA = np.cos(np.radians(measurement_data.mean_SZA.values))
    rsp_raz = measurement_data.mean_RAZ.values

    model_data = run_rtcode(working_dir, rt_dir, freeParams, wavelength, atmosphere, surface, rsp_muSZA=rsp_muSZA, rsp_raz=rsp_raz)
    ## interpolate the model data
    measurement_data_slice = measurement_data.sel(Wavelength=wavelength, method='nearest')
    model_data_interpolated = model_data.interp_like(measurement_data_slice)
    
    ## stitch the model data
    model_1d_array = data_stitcher(model_data_interpolated, index, span, [wavelength])
    if use_stdev:
        ## read the standard deviation from measurement data
        stdev_data = data_stitcher(measurement_data, index, span, [wavelength], stdev=True)
        ## calculate the residue
        residual = (model_1d_array - measurement_1d_array)/stdev_data
    else:
        residual = model_1d_array - measurement_1d_array

    # plot GIF and parameter value plot
    if 'transect' not in working_dir: #! find a better way to do this.
        plot_inversion_diagnostics(measurement_data, model_data, dataType, wavelength, freeParams, text_file_name, plot_dir) #rename this function?
    return residual

def create_synthetic_data_oil_HARP2(working_dir, rt_dir, geometries, freeParams, wavelength, atmosphere, surface, add_noise=False, **kwargs):
    from inversion_config import inversion_config_oil as config
    # noise settings
    if add_noise:
        I_unc = kwargs['I_unc']
        Rp_unc = kwargs['Rp_unc'] 
        DoLP_unc = kwargs['DoLP_unc']  

    # Calculate model data at each geometry
    SZA, VZA, RAZ = geometries 
    # print(SZA, VZA, RAZ)
    try: 
        len(SZA)
    except:
        SZA = np.array([SZA])
    try: 
        len(VZA)
    except:
        VZA = np.array([VZA])
    try: 
        len(RAZ)
    except:
        RAZ = np.array([RAZ])        

    data_path = run_rtcode(working_dir, rt_dir, freeParams, wavelength, atmosphere, surface, rsp_muSZA=0.0, rsp_raz=0.0, mono_angle=False) # use dummy vars for RSP and SZA
    data_path = data_path[0] #! oil inversion only uses 1 wavelength
    # build xarray that combines any number of SZA, RAZ, VZA and deletes .azi files with call below
    synthetic_data = call_vec_interp(working_dir, config.rt_dir, data_path, wavelength, -1*VZA, SZA, RAZ, add_noise, **kwargs) #! use negative VZAs bc RT code is opposite orientation to HARP2

    return synthetic_data


def get_residual(measurement_data, model_data_interpolated, index, span, wavelengths_active, use_stdev=True):
    # data, index, span, wavelengths_active
    measurement_1d_array = data_stitcher(measurement_data, index, span, wavelengths_active)
    model_1d_array = data_stitcher(model_data_interpolated, index, span, wavelengths_active)
    stdev_data = data_stitcher(measurement_data, index, span, wavelengths_active, stdev=True)
    if use_stdev:
        residual = (model_1d_array - measurement_1d_array)/stdev_data
    else:
        residual = (model_1d_array - measurement_1d_array)
    
    return residual

def calculate_residual_HARP2(freeParams, measurement_data, index, span, wavelength, DELP, TAUABS, water_refindx_data, use_stdev, working_dir, text_file_name, plot_dir):
    from inversion_config import inversion_config_oil as config
    switch_oil_detection = config.switch_oil_detection
    rt_dir = config.rt_dir
    dataType = config.dataType
    from atm_surf_params import atm_surf_pars_oil
    switch_I = config.switch_I
    switch_polarization = config.switch_polarization
    atmosphere = atm_surf_pars_oil.atmosphere
    surface = atm_surf_pars_oil.surface
    
    # print('Free parameter values: ')
    # for key, val in freeParams.items():
    #     if val.vary == 1:
    #         print(f'{key:<10s}: {val.value:.12f}')   

    ## run RT code to compute model data
    n_r_surf, n_i_surf = interpolate_refractive_index(wavelength, water_refindx_data)

    wavelength_um = wavelength/1000.
    atmosphere['ALAM'] = wavelength_um
    atmosphere['DELP'] = DELP
    atmosphere['TAUABS'] = TAUABS
    surface['F5'] = [n_r_surf]
    surface['F6'] = [n_i_surf]

    SZA = measurement_data.SZA.values
    VZA = measurement_data.VZA.values
    RAZ = measurement_data.RAZ.values
    geometries = SZA, VZA, RAZ
    model_data_interpolated = create_synthetic_data_oil_HARP2(working_dir, rt_dir, geometries, freeParams, wavelength, atmosphere, surface, add_noise=False)
    residual = get_residual(measurement_data, model_data_interpolated, index, span, [wavelength], use_stdev=use_stdev) 
    # print(measurement_data)
    # print(model_data_interpolated)
    # print(residual)
    # plot GIF and parameter value plot
    if 'transect' not in working_dir: #! find a better way to do this.
        plot_inversion_diagnostics(measurement_data, model_data_interpolated, dataType, wavelength, freeParams, text_file_name, plot_dir, scattering_angle=True) #rename this function?
    return residual

def oe_inversion_oil(working_dir, measurement_data, wavelength, DELP, TAUABS, water_refindx_data, freeParams, use_stdev, text_file_name, plot_dir, HARP2_data=False): #gas_abs_coeff=0.93572458
    from inversion_config import inversion_config_oil as config
    epsfcn = config.epsfcn
    ftol = config.ftol
    xtol = config.xtol
    gtol = config.gtol
    max_nfev = config.max_nfev

    ## save the current directory path
    current_dir = os.getcwd()

    ## Find index and span for data slicing
    data = measurement_data.sel(Wavelength=wavelength, method='nearest')

    n_r_surf, n_i_surf = interpolate_refractive_index(wavelength, water_refindx_data)

    if HARP2_data == False:
        index, span = get_glint_index(data, n_r_surf, freeParams['Windspeed'].value)
        # print("Glint index, span = ", index, span)
        mini = minimize(calculate_residual_oil, freeParams,
                        args=(measurement_data, index, span, wavelength, DELP, TAUABS, water_refindx_data, use_stdev, working_dir,
                        text_file_name, plot_dir), epsfcn=epsfcn, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, calc_covar=True, nan_policy="omit")
        #minimizer_object = Minimizer(calculate_residual_oil, freeParams, fcn_args=(measurement_data, index, span,
        #wavelength, DELP, TAUABS, water_refindx_data, use_stdev, working_dir, text_file_name, plot_dir), nan_policy='omit')

        # dogbox method
        #mini = minimizer_object.least_squares(method="dogbox", diff_step = [0.001, 0.00001, 0.01], ftol=ftol, xtol=xtol ,max_nfev=max_nfev, x_scale='jac', tr_solver='lsmr')

        # trf method
        #mini = minimizer_object.least_squares(method="trf", diff_step = [0.001, 0.00001, 0.01], ftol=ftol, xtol=xtol ,max_nfev=max_nfev, tr_solver="lsmr")

        # lm method, must remove min and max from freeParams
        #mini = minimizer_object.least_squares(method="lm", diff_step=0.01, ftol=ftol, xtol=xtol, max_nfev=max_nfev)

        print(mini.params.pretty_print())
        MODRSPFILELIST = ['{:.0f}model'.format(wavelength)]
        model_data_path = working_dir + MODRSPFILELIST[0] + '.rsp'
        model_data = readModel(model_data_path, wavelength)
        return mini, model_data, model_data_path
    else:
        index, span = get_glint_index(data, n_r_surf, freeParams['Windspeed'].value, HARP2=True)
        # print("Glint index, span = ", index, span)
        # print(freeParamsz)
        mini = minimize(calculate_residual_HARP2, freeParams,
                        args=(measurement_data, index, span, wavelength, DELP, TAUABS, water_refindx_data, use_stdev, working_dir,
                        text_file_name, plot_dir), epsfcn=epsfcn, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, calc_covar=True, nan_policy="omit")
        print(mini.params.pretty_print())
        return mini
    
def run_rtcode(working_dir, rt_dir, freeParams, wavelength, atmosphere, surface, rsp_muSZA, rsp_raz, mono_angle=True, add_noise=False, **kwargs):
    # print(rsp_muSZA, rsp_raz)
    # print(freeParams)
    if add_noise:
        I_unc = kwargs['I_unc']
        Rp_unc = kwargs['Rp_unc'] 
        DoLP_unc = kwargs['DoLP_unc'] 
    
    info_dir = check_info_dir(working_dir, rt_dir)

    ## call
    modelFileName = "TESTMODEL.info"
    MODSRFFILELIST = ['ModelSurf{:.0f}'.format(wavelength)]
    MODRSPFILELIST = ['{:.0f}model'.format(wavelength)]
    wavelength_nm = wavelength/1000.
    if atmosphere['ALBEDO'] < 0:
            if freeParams["NRsrf"].vary or freeParams["NIsrf"].vary or freeParams["Windspeed"].vary:
                srfFileNameInp = "srfmod{:.0f}.txt".format(wavelength)
                windspeed = freeParams["Windspeed"].value

                Srf(MCAP=25,F1=surface['F1'], F2=surface['F2'], F3=surface['F3'], F4=surface['F4'], F5=freeParams["NRsrf"].value, F6=surface['F6'][0],
                    F7=windspeed, F8=surface['F8'], ALAM=wavelength_nm, srfFileName=MODSRFFILELIST[0], MU_SZA=rsp_muSZA).createSrf(working_dir+srfFileNameInp)
                call_vec_srf(srfFileNameInp, working_dir)

    # setup atmospheric parameters and the main input file
    if freeParams['NZc'].vary == True and freeParams['NZf'].vary == True:
        NZITEMS_mod = [[freeParams["NZf"].value, 0.0, 0.0, 0.0], [freeParams["NZc"].value, 0.0, 0.0, 0.0]]
    else:
        NZITEMS_mod = [[freeParams["NZc"].value, 0.0, 0.0, 0.0]]

    atmo = Atmo(MCAP=90, A=atmosphere['A'], B=atmosphere['B'], phi=rsp_raz, ALAM=[wavelength_nm], NR=atmosphere['NR'],
                NI=atmosphere['NI'], NZITEMS=NZITEMS_mod, NSD=atmosphere['NSD'], R1=atmosphere['R1'], R2=atmosphere['R2'],
                DELP=atmosphere['DELP'], ALBEDO=atmosphere['ALBEDO'], TAUABS=atmosphere['TAUABS'], FILESURF=MODSRFFILELIST, FILEOUT=MODRSPFILELIST, mu0=rsp_muSZA)
  
    atmo.createAtmo(info_dir+modelFileName)

    ## call RT code
    call_vec_gen_obs(modelFileName, working_dir)
    ## read model output data and concatenate model output from different wavelengths
    if mono_angle == True:
        model_data_path = working_dir + MODRSPFILELIST[0] + '.rsp'
        if not add_noise:
            model_data = readModel(model_data_path, wavelength)
        else:
            model_data = readModel(model_data_path, wavelength, add_noise=True, I_unc=I_unc, Rp_unc=Rp_unc, DoLP_unc=DoLP_unc)
        return model_data
    else:
        return MODRSPFILELIST

def call_vec_srf(info_file:str, rt_dir):

    charset = sys.stdin.encoding
    if charset is None:
        charset = 'utf8'

    os.chdir(rt_dir)
    # print(f"Calling Vec Srf on {info_file}")
    # os.system(f"./vec_srf {info_file} {kernel_flag}")
    # os.system(f"./vec_srf {info_file} 0")
    call_cmd = f"./vec_srf {info_file}"
    p = subprocess.Popen(call_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return_code = p.poll()
    out = stdout.decode(charset) # noqa
    err = stderr.decode(charset)
    # print(out.strip()) # uncomment if need screen outputs
    # print(err.strip()) # uncomment if need screen outputs
    if return_code != 0:
        err_mesg = err.strip()
        print(err_mesg)
        raise ValueError
    p.wait()

    return

def call_vec_interp(working_dir, rt_dir, rt_code_output, wavelength, VZA, SZA, RAZ, add_noise, **kwargs):
    if add_noise:
        I_unc = kwargs['I_unc']
        Rp_unc = kwargs['Rp_unc'] 
        DoLP_unc = kwargs['DoLP_unc'] 
    
    charset = sys.stdin.encoding 
    if charset is None:
        charset = 'utf8'
    
    ## make fortran formatted file containing SZA and RAZ for each view direction ##
    file_to_interpolate = f'{working_dir}{rt_code_output}'
    input_file = open(f'{working_dir}/vec_interp_input.txt', "w")
    fA120 = ffFRW('A120')
    fF105 = ffFRW('F10.5')
    input_file.write(f'{fA120.write([file_to_interpolate])}\n')
    for i in range(len(SZA)):
        input_file.write(f' SZA={fF105.write([SZA[i]])} RAZ={fF105.write([RAZ[i]])}\n' ) #name change to _SZA
    input_file.close()
        
    ## call vec_interp ##
    os.chdir(rt_dir)
    call_cmd = f"./vec_interp {working_dir}/vec_interp_input.txt"
    p = subprocess.Popen(call_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return_code = p.poll()
    out = stdout.decode(charset) # analysis:ignore
    err = stderr.decode(charset)   
    # print(out.strip()) # uncomment if need screen outputs
    # print(err.strip()) # uncomment if need screen outputs
    if return_code != 0:
        err_mesg = err.strip()
        print(err_mesg)
        raise ValueError    
    p.wait()

    ### create xarray with interpolated files ###
    view_direction_data = []
    azi_files = sorted(glob.glob(f'{file_to_interpolate}*.azi'))
    for i in range(len(azi_files)):
        
        ## read each .azi file ##
        if add_noise:
            interpolated_data_mono = readModel(azi_files[i], wavelength, add_noise=add_noise, I_unc=I_unc, Rp_unc=Rp_unc, DoLP_unc=DoLP_unc)
        else:
            interpolated_data_mono = readModel(azi_files[i], wavelength)
        
        ## interpolate to the VZA angle
        interpolated_data_mono = interpolated_data_mono.interp(VZA=[VZA[i]])
        view_direction_data.append(interpolated_data_mono)
        ## remove the .azi files
        os.remove(azi_files[i])
   
    ## combine files into xarray
    if len(view_direction_data) > 1:
        interpolated_data = xr.concat(view_direction_data, dim='VZA')
    else:
        interpolated_data = view_direction_data[0]
    
    return interpolated_data

def call_vec_gen_obs(info_file:str, rt_dir):

    charset = sys.stdin.encoding
    if charset is None:
        charset = 'utf8'
        
    # print("Calling Vec Gen Obs")
    os.chdir(rt_dir)
    # os.system(f"./vec_generate_obs info/{info_file} {INTERP} {TAUREP}")
    # os.system(f"./vec_generate_obs info/{info_file} 0 1")
    call_cmd = f"./vec_generate_obs info/{info_file} 0 1"
    p = subprocess.Popen(call_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return_code = p.poll()
    out = stdout.decode(charset) # analysis:ignore
    err = stderr.decode(charset)
    # print(out.strip()) # uncomment if need screen outputs
    # print(err.strip()) # uncomment if need screen outputs
    if return_code != 0:
        err_mesg = err.strip()
        print(err_mesg)
        raise ValueError
    p.wait()

    return

def clear_plot_dir(working_dir, plot_dir, text_file_name):
    # check if plot_dir folder exists
    # if it exists: clear all images from folder, if it does not exist: create the folder
    CHECK_FOLDER = os.path.isdir(plot_dir)
    if not CHECK_FOLDER:
        os.makedirs(plot_dir)
        print("Created info directory : ", plot_dir)
    else:
        files = glob.glob(f'{plot_dir}*')
        for f in files:
            os.remove(f) # delete file in folder
    # text_file_name
    # check if the iteration values file exists
    CHECK_FILE = os.path.isfile(f'{working_dir}{text_file_name}.txt')
    if CHECK_FILE == True:
        os.remove(f'{working_dir}{text_file_name}.txt')

    return

def create_parameter_iteration_history(working_dir, text_file_name, freeParams):
    # print(working_dir)
    f = open(f'{working_dir}{text_file_name}.txt', "a+")
    col_names = [f'{key}' for key, val in freeParams.items() if val.vary==1] # add parameters being varied to  file
    for item in col_names:
        f.write(f'{item} ')
    f.close() # close the file

    return

def check_info_dir(working_dir, rt_dir):
    ## check working folder existence and copy executables to the working folder
    CHECK_FOLDER = os.path.isdir(working_dir)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
       os.makedirs(working_dir)
       print("Created working directory: ", working_dir)
    else:
        pass 
    if not working_dir.endswith(os.path.sep):
        working_dir += os.path.sep  
    # Now copy the executables to the working directory
    if not rt_dir.endswith(os.path.sep):
        rt_dir += os.path.sep    
    src = rt_dir + 'vec_generate_obs'
    dst = working_dir + 'vec_generate_obs'
    shutil.copy2(src, dst)

    src = rt_dir + 'vec_srf'
    dst = working_dir + 'vec_srf'
    shutil.copy2(src, dst)  

    src = rt_dir + 'vec_interp'
    dst = working_dir + 'vec_interp'
    shutil.copy2(src, dst)    
    
    info_dir = working_dir + 'info/'
    CHECK_FOLDER = os.path.isdir(info_dir)
    if not CHECK_FOLDER:
       os.makedirs(info_dir)
       print("Created info directory : ", info_dir)
    else:
        pass 
    
    return info_dir


if __name__ == "__main__":
    ''' Purpose: Testing this python file. Code below does not run unless this file is executed as "main"
    '''
    start_time = time.time()

### This part tests the inversion for oil case

    # import inversion configuration
    from inversion_config import inversion_config_oil as config
    rt_root_dir = config.rt_root_dir
    freeParams = config.freeParams
    water_rfindx_path = config.water_rfindx_path
    merra2_data_dir = config.merra2_data_dir
    merra2_hdf_dir = config.merra2_hdf_dir
    merra2_hdf_path = config.merra2_hdf_path
    measurement_file_path = config.measurement_file_path
    start_scan = config.start_scan
    end_scan = config.end_scan
    dataType = config.dataType
    switch_I = config.switch_I
    layer_m = config.layer_m
    switch_polarization = config.switch_polarization
    use_stdev = config.use_stdev
    correct_gas_abs = config.correct_gas_abs

    working_dir = rt_root_dir + 'rt_code_work_test/' # must be under the root folder of rt_code otherwise some libraries won't load

    ############## set up plotting directories and files ##############
    plot_dir = working_dir + '/imagesforGIF/'
    text_file_name = 'inversion_iteration_values'
    clear_plot_dir(working_dir, plot_dir, text_file_name)
    create_parameter_iteration_history(working_dir, text_file_name, freeParams)
    ###################################################################

    water_refindx_data = read_water_refractive_index(water_rfindx_path)
    # read measurment data
    measurement_data = read_inversion_input_file(measurement_file_path, dataType, start_scan=start_scan, end_scan=end_scan, row=config.row_pixel, col=config.col_pixel)
    # print(measurement_data.RI.values)
    # Get active wavelengths from switches
    wavelengths_active = get_active_wavelengths(measurement_data.Wavelength.values, switch_I, switch_polarization)
    print('Active wavelengths are: ', wavelengths_active)
    if dataType == 1 or dataType == 2:
        RSP_altitude = measurement_data.mean_RSP_altitude.values
        layer_m.append(RSP_altitude)
    layer_m = sorted(layer_m)
    print(f'Atmosphere layers are at: {layer_m} meters.')
    if correct_gas_abs == True:
        #if not os.path.exists(merra2_hdf_path):
        #    os.system(f'python3 merra2intorsp.py {measurement_file_path} {merra2_data_dir} {merra2_hdf_dir}')
        if dataType == 1 or dataType == 2:
            hdf = SD(merra2_hdf_path, SDC.READ)
            xo3col, xno2col, xh2ocol = np.nanmean(hdf.select("TO3")[start_scan:end_scan + 1]) / 1000, config.xno2col, np.nanmean(hdf.select("MERRA2_PWV_column")[start_scan:end_scan + 1])
            DELP = altToPressure(layer_m, hdf=hdf)
        
        elif dataType == 3:
            ancillary = xr.open_dataset(config.HARP2_ancillary)
            lat, lon = measurement_data.mean_latitude.values, measurement_data.mean_longitude.values
            idx1, idx2 = np.where((ancillary.latitude.values == measurement_data.mean_latitude.values) & (ancillary.longitude.values == measurement_data.mean_longitude.values))
            xo3col, xno2col, xh2ocol = ancillary.TO3.values[idx1, idx2][0]/1000, ancillary.TOTCOL_NO2.values[idx1, idx2][0], ancillary.TQV.values[idx1, idx2][0]/10
            SP = ancillary.PS.values[idx1, idx2][0] / 100 # convert Pa to mbar (hPa), surface pressure
            DELP = [200., 250., 275., SP-(200.+250.+275.)] # make an atmosphere containining 4 layers that add up to the surface pressure
            print(f'WS: {np.sqrt(ancillary.U10M.values[idx1, idx2]**2 + ancillary.V10M.values[idx1, idx2]**2)} ')
    
    else:
        DELP = altToPressure(layer_m)
    print(f'DELP: {DELP}')
    
    for wavelength in wavelengths_active:
        if correct_gas_abs == True:  # ! only works for one wavelength at a time
            TAUABS = absorption_profile(DELP, xo3col, xno2col, xh2ocol, wavelength)
        else:
            TAUABS = [[]]
        print('Now retrieving for wavelength: ', wavelength, ' nm')
        if dataType != 3:
            mini, model_data, model_data_path = oe_inversion_oil(working_dir, measurement_data, wavelength, DELP, TAUABS, water_refindx_data, freeParams, use_stdev, text_file_name, plot_dir)
        else:
            mini = oe_inversion_oil(working_dir, measurement_data, wavelength, DELP, TAUABS, water_refindx_data, freeParams, use_stdev, text_file_name, plot_dir, HARP2_data=True) 
    ########## create GIF and inversion iteration plot ################ 
    create_GIF(plot_dir, working_dir, 'inversion_gif') 
    create_inversion_netcdf(mini, f'{working_dir}{text_file_name}.txt', working_dir, 'inversion_results')
    inv_iteration_plot(f'{working_dir}{text_file_name}.txt', working_dir, 'inv_iteration_plot')    
    
    print("--- %s seconds ---" % (time.time() - start_time))