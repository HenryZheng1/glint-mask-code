# MODIFY PYTHON PATH TO INCLUDE DIRECTORY
from __future__ import print_function
from typing import List
import h5py 
from reader.read_HARP2 import get_HARP2_variable
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import pandas as pd

def coxmunk(DMUR, CN1, CN2, WS, PHIW, DMUI, PHII, PHIR, Gaussian):
    # **************************************************************
    #   TRANSLATED FROM MISHCHENKO'S FORTRAN SUBROUTINE,
    #   THIS IDL FUNCTION CALCULATES THE FRESNEL REFLECTION MATRIX
    #   FOR THE OCEAN SURFACE (SEE ORIGINAL COMMENT BELOW)
    #   ADDING THE GRAM-CHARLIER DISTRIBUTION.
    #	NOTE THAT THIS ARGUMENT LIST ACCEPTS WIND SPEED IN m/s AND NOT s**2
    #
    #
    #                                      M.OTTAVIANI 2009
    # --------------------------------------------------------------
    #   ---ORIGINAL COMMENT
    #
    #   CALCULATION OF THE STOKES REFLECTION MATRIX FOR
    #   ILLUMINATION FROM ABOVE FOR A REFRACTIVE MEDIUM WITH
    #   A STATISTICALLY ROUGH SURFACE SEPARATING TWO HALF-SPACES
    #   WITH REFRACTIVE INDICES OF THE UPPER AND LOWER HALF-SPACES EQUAL TO
    #   CN1 AND CN2, RESPECTIVELY.
    #   SIGMA2 = MEAN SQUARE SURFACE SLOPE (S**2 in Tsang et al., page 87)
    #   DMUI = ABS(COSINE OF THE INCIDENT ZENITH ANGLE)
    #   PHII = INCIDENT AZIMUTH ANGLE.
    #   DMUR = ABS(COSINE OF THE REFLECTION ZENITH ANGLE).
    #   PHIR = REFLECTION AZIMUTH ANGLE
    #   R - (4X4) REFLECTION MATRIX
    #
    # **********************************************************
    #
    #   CARTESIAN COMPONENTS OF THE UNIT VECTORS OF THE INCIDENT AND
    #   SCATTERED BEAMS
    #
    if (np.abs(DMUI - np.float64(1.0)) < np.float64(1e-9)):
        DMUI = np.float64(0.999999999999999)

    if (np.abs(DMUR - np.float64(1.0)) < np.float64(1e-9)):
        DMUR = np.float64(0.999999999999999)

    DCOSI = np.float64(np.cos(PHII))
    DSINI = np.float64(np.sin(PHII))
    DCOSR = np.float64(np.cos(PHIR))
    DSINR = np.float64(np.sin(PHIR))
    DSI = np.sqrt(np.float64(1.0) - DMUI*DMUI)
    DSR = np.sqrt(np.float64(1.0) - DMUR*DMUR)
    VI1 = DSI*DCOSI
    VI2 = DSI*DSINI
    VI3 = -DMUI
    VR1 = DSR*DCOSR
    VR2 = DSR*DSINR
    VR3 = DMUR

#
#    LOCAL SURFACE NORMAL FOR SPECULAR REFLECTION
#
    UNIT1 = VI1 - VR1
    UNIT2 = VI2 - VR2
    UNIT3 = VI3 - VR3
    FACT1 = UNIT1*UNIT1 + UNIT2*UNIT2 + UNIT3*UNIT3
    FACTOR = np.sqrt(np.float64(1)/FACT1)

#
#    FRESNEL REFLECTION COEFFICIENTS
#
    XI1 = FACTOR * (UNIT1*VI1 + UNIT2*VI2 + UNIT3*VI3)
    CXI2 = np.float64(1) - (np.float64(1) - XI1*XI1)*CN1*CN1 / (CN2*CN2)
    CXI2 = np.sqrt(CXI2)
    C1 = CN1*XI1
    C2 = CN2*CXI2
    CRPER = (C1-C2)/(C1+C2)
    C1 = CN2*XI1
    C2 = CN1*CXI2
    CRPAR = (C1-C2)/(C1+C2)

#
#    CALCULATION OF THE AMPLITUDE SCATTERING MATRIX
#
    TI1 = -DMUI*DCOSI
    TI2 = -DMUI*DSINI
    TI3 = -DSI

    TR1 = DMUR*DCOSR
    TR2 = DMUR*DSINR
    TR3 = -DSR

    PI1 = -DSINI
    PI2 = DCOSI
    PI3 = np.float64(0.0)

    PR1 = -DSINR
    PR2 = DCOSR
    PR3 = np.float64(0.0)

    PIKR = PI1*VR1 + PI2*VR2 + PI3*VR3
    PRKI = PR1*VI1 + PR2*VI2 + PR3*VI3
    TIKR = TI1*VR1 + TI2*VR2 + TI3*VR3
    TRKI = TR1*VI1 + TR2*VI2 + TR3*VI3

    E1 = PIKR*PRKI
    E2 = TIKR*TRKI
    E3 = TIKR*PRKI
    E4 = PIKR*TRKI

    CF11 = E1*CRPER + E2*CRPAR
    CF12 = -E3*CRPER + E4*CRPAR
    CF21 = -E4*CRPER + E3*CRPAR
    CF22 = E2*CRPER + E1*CRPAR

#
#   CALCULATION OF THE STOKES REFLECTION MATRIX
#
    VP1 = VI2*VR3 - VI3*VR2
    VP2 = VI3*VR1 - VI1*VR3
    VP3 = VI1*VR2 - VI2*VR1

    DMOD = VP1*VP1 + VP2*VP2 + VP3*VP3
    DMOD = DMOD*DMOD

    RDZ2 = UNIT3*UNIT3
    RDZ4 = RDZ2*RDZ2

    if (Gaussian == False):
        SIGMA2U = 0.00316*WS
        SIGMA2C = 0.003 + 0.00192*WS
        SIGMAU = np.sqrt(SIGMA2U)
        SIGMAC = np.sqrt(SIGMA2C)

#   CARTESIAN COMPONENTS OF WAVE VECTORS ALONG/ACROSS WIND DIRECTIONS

        DCOSIW = np.cos(PHII-PHIW)
        DSINIW = np.sin(PHII-PHIW)
        DCOSRW = np.cos(PHIR-PHII-PHIW)
        DSINRW = np.sin(PHIR-PHII-PHIW)

        VIW1 = DSI*DCOSIW
        VIW2 = DSI*DSINIW
        VIW3 = -DMUI
        VRW1 = DSR*DCOSRW
        VRW2 = DSR*DSINRW
        VRW3 = DMUR

        UNIT1W = -(VIW1-VRW1)
        UNIT2W = -(VIW2-VRW2)
        UNIT3W = -(VIW3-VRW3)
        #FACT1W = UNIT1W*UNIT1W + UNIT2W*UNIT2W + UNIT3W*UNIT3W
        #FACTORW = np.sqrt(np.float64(1)/FACT1W)

# Gram-Charlier distribution

        CSI = UNIT2W / (UNIT3W*SIGMAC)  # Crosswind component
        ETA = UNIT1W / (UNIT3W*SIGMAU)  # Upwind component
        C21 = 0.01 - 0.0086*WS
        C03 = 0.04 - 0.033*WS
        C40 = 0.40
        C22 = 0.12
        C04 = 0.23

        GCP = np.float64(1) - (C21/2.) * (CSI**2 - 1.)*ETA - \
            (C03/6.) * (ETA**2 - 3.)*ETA + \
            (C40/24.) * (CSI**4 - 6.*CSI**2 + 3) + \
            (C22/4.) * (CSI**2 - 1)*(ETA**2 - 1) + \
            (C04/24.) * (ETA**4 - 6.*ETA**2 + 3)

        DCOEFF = np.float64(1.0)/(np.float64(4.0)*DMUI *
                                  DMUR*DMOD*RDZ4*np.float64(2.0)*(SIGMAC*SIGMAU))
        DEX = -(CSI*CSI + ETA*ETA)/np.float64(2.0)
        DEX = np.exp(DEX)
        DCOEFF = DCOEFF*FACT1*FACT1*DEX*GCP

    else:
        SIGMA2 = 0.5*(0.003 + 0.00512*WS)  # .003*.00512*WS
        DCOEFF = np.float64(1.0)/(np.float64(4.0)*DMUI *
                                  DMUR*DMOD*RDZ4*np.float64(2.0)*SIGMA2)
        DEX = -(UNIT1*UNIT1 + UNIT2*UNIT2)/(np.float64(2.0)*SIGMA2*RDZ2)
        DEX = np.exp(DEX)
        DCOEFF = DCOEFF*FACT1*FACT1*DEX
    # print(UNIT1, UNIT2, UNIT3)
    AF = 0.5*DCOEFF
    AF11 = np.abs(CF11)
    AF12 = np.abs(CF12)
    AF21 = np.abs(CF21)
    AF22 = np.abs(CF22)

    AF11 = AF11*AF11
    AF12 = AF12*AF12
    AF21 = AF21*AF21
    AF22 = AF22*AF22

    R = np.zeros((4, 4), dtype='complex')
 
    R[0, 0] = (AF11 + AF12 + AF21 + AF22)*AF
    R[0, 1] = (AF11 - AF12 + AF21 - AF22)*AF
    R[1, 0] = (AF11 - AF22 + AF12 - AF21)*AF
    R[1, 1] = (AF11 - AF12 - AF21 + AF22)*AF

    CI = np.vectorize(complex)(np.float64(0.0), np.float64(-1.0))

    C21 = np.conj(CF21)
    C22 = np.conj(CF22)
    CTTTP = CF11*np.conj(CF12)
    CTTPT = CF11*C21
    CTTPP = CF11*C22
    CTPPT = CF12*C21
    CTPPP = CF12*C22
    CPTPP = CF21*C22

    R[0, 2] = (-CTTTP-CPTPP)*DCOEFF
    R[0, 3] = -CI*(CTTTP+CPTPP)*DCOEFF
    R[1, 2] = (-CTTTP+CPTPP)*DCOEFF
    R[1, 3] = -CI*(CTTTP-CPTPP)*DCOEFF
    R[2, 0] = (-CTTPT-CTPPP)*DCOEFF
    R[2, 1] = (-CTTPT+CTPPP)*DCOEFF
    R[2, 2] = (CTTPP+CTPPT)*DCOEFF
    R[2, 3] = CI*(CTTPP-CTPPT)*DCOEFF
    R[3, 0] = CI*(CTTPT+CTPPP)*DCOEFF
    R[3, 1] = CI*(CTTPT-CTPPP)*DCOEFF
    R[3, 2] = -CI*(CTTPP+CTPPT)*DCOEFF
    R[3, 3] = (CTTPP-CTPPT)*DCOEFF

    #R = np.real(R)

    return R

def getDoLP_CM(VZA:List[float], AZI:List[float], SZA:float, SRI:float, WS:float): 
    """Returns the Cox and and Munk DoLP

    Args:
        VZA (List[float]): List of Viewing zenith angles
        AZI (List[float]): List of Azimuths
        SZA (float): Solar Zenith Angle in degrees
        SRI (float): Surface Refractive index
        WS (float): Windspeed m/s

    Returns:
        List[float]: DoLP Cox and Munk
    """    
    RM = np.zeros((4, 4, len(VZA)))
    for i in range(len(VZA)):   
        if VZA[i] >= 0:
            PHIR = np.deg2rad(AZI) 
        else:
            PHIR = np.deg2rad(AZI) - np.pi
        R = coxmunk(abs(np.cos(np.deg2rad(VZA[i]))), 1.0, SRI, WS, 0, np.cos(np.deg2rad(SZA)), 0, PHIR, True)
        RM[:, :, i] = np.real(R)
    DoLP_CM = np.sqrt(RM[0, 1, :]**2 + RM[0, 2, :]**2) / RM[0, 0, :]
    return DoLP_CM


def findLIS(A, n):
    hash = dict()
    LIS_size, LIS_index = 1, 0

    hash[A[0]] = 1
    LIS_list = []
    for i in range(1, n):
        if A[i] - 1 not in hash:
            hash[A[i] - 1] = 0

        hash[A[i]] = hash[A[i] - 1] + 1
        if LIS_size < hash[A[i]]:
            LIS_size = hash[A[i]]
            LIS_index = A[i]

    start = LIS_index - LIS_size + 1
    while start <= LIS_index:
        # print(start, end=",")
        LIS_list.append(start)
        start += 1

    return LIS_list


#NOTE: this methods hasn't been reviewed, pls do not implement for the time being
# this is the original getGlintRegion
def getGlintRegion(DoLP, DoLP_CM, VZA, specularidx, DoLP_threshold = 0.005, nangles_min = 40, VZA_threshold = 20.):
    DoLP_Diff = np.abs(DoLP - DoLP_CM)
    # Sunglint is where the DoLP is within DoLP_thresold from that predicted by the Cox - Munk distribution
    glint_region = np.where((DoLP_Diff <= DoLP_threshold) & (np.abs(VZA-VZA[specularidx]) <= VZA_threshold))
    glint_region_angles = VZA[glint_region]
    nGlintAngles = len(glint_region_angles)
    # print(nGlintAngles)
    # print(glint_region)

    while nGlintAngles < nangles_min:
        DoLP_threshold += 0.005
        nangles_min += -1
        glint_region = np.where((DoLP_Diff <= DoLP_threshold) & (np.abs(VZA-VZA[specularidx]) <= VZA_threshold))
        glint_region_angles = VZA[glint_region]
        nGlintAngles = len(glint_region_angles)

    glint_region = np.reshape(glint_region, (nGlintAngles,))
    glint_region = findLIS(np.transpose(glint_region), len(glint_region))
    index, span = np.nanmin(glint_region), np.nanmax(glint_region)
    return index, span

# this function pulls variables for all pixels of a HAPR2 
def get_wl(file_path, wavelength): #! rename
        HARP2 = h5py.File(file_path, "r")
        intensity_F0 = get_HARP2_variable(HARP2, 'intensity_F0', wavelength)[:]
        # pol_F0 = get_HARP2_variable(HARP2, 'polarization_F0', wavelength)[:]
        # sun_earth_dist = HARP2.attrs['sun_earth_distance']
        # sun_earth_dist  = 1.50*10**11 # r_0 (mean earth sun distance 1.5 10^11 m)
        sun_earth_dist = 0.990849042172323 # AU

        ## define orbital geometries
        SZA  = get_HARP2_variable(HARP2, 'solar_zenith_angle', wavelength)[:]
        # print(SZA.shape)
        # mean_SZA = np.nanmean(SZA)
        # scatang = get_HARP2_variable(HARP2, 'scattering_angle', wavelength)[:]
        VAA = get_HARP2_variable(HARP2, 'sensor_azimuth_angle', wavelength)[:]
        SAA = get_HARP2_variable(HARP2, 'solar_azimuth_angle', wavelength)[:]
        RAZ = SAA - VAA
        # mean_RAZ = np.nanmean(RAZ)

        RI = get_HARP2_variable(HARP2, 'i', wavelength)[:]
        # print(RI.shape, SZA.shape, intensity_F0.shape)
        RI = ((np.pi * (sun_earth_dist**2) * RI) / (intensity_F0 * np.cos(np.deg2rad(SZA))))
        RI_stdev = get_HARP2_variable(HARP2, 'i_stdev', wavelength)[:]
        RI_stdev = np.tile(RI_stdev, (1, 1)) 

        DoLP = get_HARP2_variable(HARP2, 'dolp', wavelength)[:]
        # DoLP_stdev = U_stdev = get_HARP2_variable(HARP2, 'dolp_stdev', wavelength)[:]

        VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', wavelength)[:]
        
        ## altitude = get_HARP2_variable(HARP2, 'altitude', wavelength)[row, col] #! not in dataset
        # longitude = get_HARP2_variable(HARP2, 'longitude', wavelength)[:]
        # latitude = get_HARP2_variable(HARP2, 'latitude', wavelength)[:]
        return SZA, VZA, RAZ, RI, DoLP

def DOLP_CM_HARP2(RAZ, VZA, SZA, SRI, WS):
    DoLP_CM = []
    for i in range(len(RAZ)):
        VZA_array = np.array([VZA[i]])
        RAZ_array = np.array([RAZ[i]-180.]) #!!!!!
        SZA_array = np.array([SZA[i]])
        DoLP_CM.append(getDoLP_CM(VZA_array, RAZ_array, SZA_array, SRI, WS)[0])
    DoLP_CM = np.array(DoLP_CM)
    return DoLP_CM

def get_critical_angle(SZA, VZA, RAZ): #! rename
    theta_s = np.deg2rad(SZA)
    theta_v = np.deg2rad(VZA)
    phi_r = np.deg2rad(RAZ)

    arg = (np.cos(theta_s)*np.cos(theta_v)) - (np.sin(theta_s)*np.sin(theta_v)*np.cos(phi_r))
    return np.rad2deg(np.arccos(arg))

# this was an attrempt to update the code to use the critical angle instead of the specular index
def getGlintRegion_update(DoLP, DoLP_CM, critical_angle, DoLP_threshold = 0.005, nangles_min = 40, angle_threshold=20.):
    ## set up conditions based on the strenght of the glint 
    strongest_glint_angle = np.nanmin(critical_angle) # sunglint is strongest where the critical angle is the smallest
    
    # Sunglint is where the DoLP is within DoLP_thresold from that predicted by the Cox - Munk distribution
    DoLP_Diff = np.abs(DoLP - DoLP_CM)
    angle_diff = np.abs(critical_angle - strongest_glint_angle)
    
    glint_region = np.where((DoLP_Diff <= DoLP_threshold) & (angle_diff <= angle_threshold))
    glint_region_angles = critical_angle[glint_region]
    nGlintAngles = len(glint_region_angles)

    while nGlintAngles < nangles_min:
        DoLP_threshold += 0.005
        nangles_min += -1
        angle_threshold += -1
        glint_region = np.where((DoLP_Diff <= DoLP_threshold) & (angle_diff <= angle_threshold))
        glint_region_angles = critical_angle[glint_region]
        nGlintAngles = len(glint_region_angles)

    glint_region = np.reshape(glint_region, (nGlintAngles,))
    glint_region = findLIS(np.transpose(glint_region), len(glint_region))
    index, span = np.nanmin(glint_region), np.nanmax(glint_region)
    return index, span
      

if __name__ == '__main__':
    import time
    from inversion_oil import interpolate_refractive_index, read_water_refractive_index
    from inversion_config import inversion_config_oil as config
    
    data_dir = config.data_dir
    measurement_file_path = data_dir + config.measurement_file_name
    start_scan, end_scan = 2800, 2819 #280, 300 #500-520
    wl = 867.
    import xarray as xr
    from HARP2_oil_retrieval import get_pixel_nums
    
    st = time.time()
    # plt.subplots_adjust(wspace=0.1, hspace=0.25)
    row, col = get_pixel_nums(measurement_file_path)
    # print(row, col)
    # row, col = 30, 30
    water_refindx_data = read_water_refractive_index(config.water_rfindx_path)
    WS = 5.0 
    SRI_867, n_i_surf = interpolate_refractive_index(867., water_refindx_data)
    # print(SRI_867)
    SRI_670, n_i_surf = interpolate_refractive_index(670., water_refindx_data)
    angles_670 = np.full((row, col), fill_value=np.nan)
    angles_867 = np.full((row, col), fill_value=np.nan)
    
    path = '/Users/olivia/Downloads/code/snow_code/data/PACE/PACE.20220321T183042.L1C.5.2km.ANC.nc'
    ancillary = xr.open_dataset(path)
    waterfraction = ancillary.waterfraction.values

    HARP2 = h5py.File(measurement_file_path, "r")
    longitude = get_HARP2_variable(HARP2, 'longitude', 867)[:]
    latitude = get_HARP2_variable(HARP2, 'latitude', 867)[:]

    SZA_867, VZA_867, RAZ_867, RI_867, DoLP_867 = get_wl(measurement_file_path, 867)
    SZA_670, VZA_670, RAZ_670, RI_670, DoLP_670 = get_wl(measurement_file_path, 670)

    VZA_867_switch = get_HARP2_variable(HARP2, 'view_angles', 867)
    VZA_670_switch = get_HARP2_variable(HARP2, 'view_angles', 670)

    # define critical angle
    # find min and find angle away
    # flag the pixels
    plot_path = '/Users/olivia/Downloads/code/snow_code/data/PACE/granT1830/'
    ## plots the RI, DoLP, DoLP diff for each pixel ###
    for r in np.arange(0, row, 1):
        for c in np.arange(0, col, 1):
            print(r, c)
            if np.isnan(VZA_867[r, c]).all() == True or np.isnan(RI_867[r, c]).all() == True or waterfraction[r, c] != 1.0:
                continue
            else:
                try:
                    # print(RAZ_867[r, c], VZA_670[r, c], SZA_670[r, c])
                    DOLP_CM_670 = DOLP_CM_HARP2(RAZ_670[r, c], VZA_670[r, c], SZA_670[r, c], SRI_670, WS)
                    DOLP_CM_867 = DOLP_CM_HARP2(RAZ_867[r, c], VZA_867[r, c], SZA_867[r, c], SRI_867, WS)

                    critical_angle_867 = get_critical_angle(SZA_867[r, c], VZA_867[r, c], RAZ_867[r, c])
                    critical_angle_670 = get_critical_angle(SZA_670[r, c], VZA_670[r, c], RAZ_670[r, c])
                    print('CRIT:', critical_angle_670)
                    print('RAZ', np.round(RAZ_670[r, c]))
                    print('VZA', np.round(VZA_670[r, c]))
                    print('SZA', np.round(SZA_670[r, c]))
                    
                    index_670, span_670 = getGlintRegion_update(DoLP_670[r, c], DOLP_CM_670, critical_angle_670, nangles_min=20)

                    print('CRIT:', np.round(critical_angle_867))
                    print('RAZ', np.round(RAZ_867[r, c]))
                    print('VZA', np.round(VZA_867[r, c]))
                    print('SZA', np.round(SZA_867[r, c]))
                    index_867, span_867 = getGlintRegion_update(DoLP_867[r, c], DOLP_CM_867, critical_angle_867, nangles_min=8)

                    fig, axs = plt.subplots(3, 1)
                    plt.subplots_adjust(hspace=0.25)

                    axs[0].plot(VZA_670_switch, RI_670[r, c], 'o', color='green', label='670')
                    axs[1].plot(VZA_670_switch, DoLP_670[r, c], 'o', color='green')
                    axs[1].plot(VZA_670_switch, DOLP_CM_670, '-', color='black')

                    axs[2].plot(VZA_670_switch, (DoLP_670[r, c]-DOLP_CM_670), '-', color='green')
                    axs[2].plot(VZA_867_switch, (DoLP_867[r, c]-DOLP_CM_867), '-', color='purple')

                    axs[1].axvspan(VZA_670_switch[index_670], VZA_670_switch[span_670],alpha=0.5, color='green')
                    axs[0].axvspan(VZA_670_switch[index_670], VZA_670_switch[span_670], alpha=0.5, color='green')

                    axs[0].plot(VZA_867_switch, RI_867[r, c], 'o', color='purple', label='867')
                    axs[1].plot(VZA_867_switch, DoLP_867[r, c], 'o', color='purple')
                    axs[1].plot(VZA_867_switch, DOLP_CM_867, '-', color='red')
                    axs[1].axvspan(VZA_867_switch[index_867], VZA_867_switch[span_867], alpha=0.5, color='purple')
                    axs[0].axvspan(VZA_867_switch[index_867], VZA_867_switch[span_867], alpha=0.5, color='purple')
                    
                    for i in range(len(critical_angle_867)):
                        axs[0].annotate(str(round(critical_angle_867[i])), (VZA_867_switch[i], RI_867[r, c, i]+0.005))

                    axs[0].tick_params(axis='x', colors='green')
                    axs[1].tick_params(axis='x', colors='green')
                    axs[0].set_xlabel('VZA')
                    axs[-1].set_xlabel('VZA')
                    axs[0].set_title(f'({r}, {c})')
                    axs[-1].axhline(0.0)
                    axs[-1].set_ylim(-0.01, 0.01)
                    # plt.show()
                except IndexError as err:
                    print(err)
                    # print(f'{r}, {c} failed')
                    continue
                except ValueError as err:
                    print(err)
                    # print(f'{r}, {c} failed')
                    continue
                # plt.show()
                
                plt.savefig(f'{plot_path}/pixel_{r}_{c}.png')
                plt.close()