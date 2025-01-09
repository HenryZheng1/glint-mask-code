import os, sys, warnings, h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
from reader.read_PM import get_raz
warnings.filterwarnings("ignore")
import cProfile
# from reader.read_POLDER import get_scattering_angle

def split_wavelengths(data_dictionary, var, wavelength):
    if len(data_dictionary.keys()) == 9:
        wavelengths = data_dictionary['sensor_views_bands']['intensity_wavelength'][:, 0]
    else:
        wavelengths = data_dictionary['intensity_wavelength'][:, 0]
    unique_wavelengths = np.unique(wavelengths) #[441.9, 549.8, 669.4, 867.8]
    if var.shape[0] == 90:
        if wavelength == 442:
            var = var[np.where(wavelengths == unique_wavelengths[0])[0]]
        elif wavelength == 550:
            var = var[np.where(wavelengths == unique_wavelengths[1])[0]]
        elif wavelength == 670:
            var = var[np.where(wavelengths == unique_wavelengths[2])[0]]
        elif wavelength == 867:
            var = var[np.where(wavelengths == unique_wavelengths[3])[0]]
    else:
        if wavelength == 442:
            var = var[:, :, np.where(wavelengths == unique_wavelengths[0])[0]]
        elif wavelength == 550:
            var = var[:, :, np.where(wavelengths == unique_wavelengths[1])[0]]
        elif wavelength == 670:
            var = var[:, :, np.where(wavelengths == unique_wavelengths[2])[0]]
        elif wavelength == 867:
            var = var[:, :, np.where(wavelengths == unique_wavelengths[3])[0]]

    return var
    
def get_HARP2_variable(data_dictionary, SDS_variable, wavelength):
    for key in data_dictionary.keys():
        if len(data_dictionary.keys()) == 9:
            if key == 'geolocation_data' or key=='sensor_views_bands' or key=='bin_attributes' or key=='observation_data':
                if SDS_variable in data_dictionary[key].keys():
                    var = data_dictionary[key][SDS_variable][:]
                    # print(var)
                    fill_val = data_dictionary[key][SDS_variable].attrs['_FillValue']
                    # print(fill_val)
                    var = np.where(var==fill_val, np.nan, var)
                    # print(var)
                    if 'scale_factor' in data_dictionary[key][SDS_variable].attrs.keys():
                        offset = data_dictionary[key][SDS_variable].attrs['add_offset']
                        scale_factor = data_dictionary[key][SDS_variable].attrs['scale_factor']
                        var = (var * scale_factor) + offset
                    if len(var.shape) == 4:
                        return split_wavelengths(data_dictionary, var[:, :, :, 0], wavelength)
                    elif len(var.shape) == 3:
                        return split_wavelengths(data_dictionary, var, wavelength)
                    elif len(var.shape) == 2 and var.shape[0] == 90:
                        return split_wavelengths(data_dictionary, var[:, 0], wavelength)
                    elif len(var.shape) == 1 and var.shape[0] == 90:
                        return split_wavelengths(data_dictionary, var, wavelength)
                    else:
                        return var
        else:
            var = data_dictionary[SDS_variable][:]
            # print(var.shape)
            if len(var.shape) == 4:
                return split_wavelengths(data_dictionary, var[:, :, :, 0], wavelength)
            if len(var.shape) == 3 or (len(var.shape) == 1 and var.shape[0] == 90):
                return split_wavelengths(data_dictionary, var, wavelength)
            elif len(var.shape) == 2 and var.shape[0] == 90:
                return split_wavelengths(data_dictionary, var[:, 0], wavelength)
            else:
                return var

def create_HARP2_RGB(file_path):
    HARP2 = h5py.File(file_path, "r")
    # sun_earth_dist = HARP2.attrs['sun_earth_distance']
    sun_earth_dist = 0.990849042172323 # AU

    VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', 670)[:]
    min_VZA_idx = np.array(np.where(VZA == np.nanmin(VZA)))
    min_VZA_idx = int(min_VZA_idx) if len(min_VZA_idx) == 1 else min_VZA_idx[-1][0]
    R = get_HARP2_variable(HARP2, 'i', 670)[:, :, min_VZA_idx]
    intensity_F0 = get_HARP2_variable(HARP2, 'intensity_f0', 670)[0]
    SZA  = get_HARP2_variable(HARP2, 'solar_zenith_angle', 670)[:, :, min_VZA_idx]
    R = ((np.pi * (sun_earth_dist**2) * R) / (intensity_F0 * np.cos(np.deg2rad(SZA))))

    VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', 442)[:]
    min_VZA_idx = np.array(np.where(VZA == np.nanmin(VZA)))
    min_VZA_idx = int(min_VZA_idx) if len(min_VZA_idx) == 1 else min_VZA_idx[-1][0]
    B = get_HARP2_variable(HARP2, 'i', 442)[:, :, min_VZA_idx]
    intensity_F0 = get_HARP2_variable(HARP2, 'intensity_f0', 442)[0]
    SZA  = get_HARP2_variable(HARP2, 'solar_zenith_angle', 442)[:, :, min_VZA_idx]
    B = ((np.pi * (sun_earth_dist**2) * B) / (intensity_F0 * np.cos(np.deg2rad(SZA))))

    VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', 550)[:]
    min_VZA_idx = np.array(np.where(VZA == np.nanmin(VZA)))
    min_VZA_idx = int(min_VZA_idx) if len(min_VZA_idx) == 1 else min_VZA_idx[-1][0]
    G = get_HARP2_variable(HARP2, 'i', 550)[:, :, min_VZA_idx]
    intensity_F0 = get_HARP2_variable(HARP2, 'intensity_f0', 550)[0]
    SZA  = get_HARP2_variable(HARP2, 'solar_zenith_angle', 550)[:, :, min_VZA_idx]
    G = ((np.pi * (sun_earth_dist**2) * G) / (intensity_F0 * np.cos(np.deg2rad(SZA))))
    # print(R)

    image = np.dstack((R, G, B))
    plt.imshow(image)
    plt.show()
    return


def readHARP2(file_path, wavelength, row, col):
    if wavelength not in [442, 550, 670, 867]:
        raise Exception(f"Wavelength must be one of [442, 550, 670, 867] not {wavelength}.")
    HARP2 = h5py.File(file_path, "r")
    intensity_F0 = get_HARP2_variable(HARP2, 'intensity_f0', wavelength)[:]
    pol_F0 = get_HARP2_variable(HARP2, 'polarization_F0', wavelength)[:]
    # sun_earth_dist = HARP2.attrs['sun_earth_distance']
    # sun_earth_dist  = 1.50*10**11 # r_0 (mean earth sun distance 1.5 10^11 m)
    sun_earth_dist = 1.005954 # AU

    ## define orbital geometries
    SZA  = get_HARP2_variable(HARP2, 'solar_zenith_angle', wavelength)[row, col, :]
    # print(SZA.shape)
    mean_SZA = np.nanmean(SZA)
    scatang = get_HARP2_variable(HARP2, 'scattering_angle', wavelength)[row, col, :]
    VAA = get_HARP2_variable(HARP2, 'sensor_azimuth_angle', wavelength)[row, col, :]
    SAA = get_HARP2_variable(HARP2, 'solar_azimuth_angle', wavelength)[row, col, :]
    RAZ = SAA - VAA #! RAZ correct?
    # RAZ = get_raz(SAA, VAA)
    mean_RAZ = np.nanmean(RAZ)

    RI = get_HARP2_variable(HARP2, 'i', wavelength)[row, col, :]
    # print(RI.shape, SZA.shape, intensity_F0.shape)
    RI = ((np.pi * (sun_earth_dist**2) * RI) / (intensity_F0 * np.cos(np.deg2rad(SZA))))
    RI = np.tile(RI, (1, 1)) 
    RI_stdev = get_HARP2_variable(HARP2, 'i_stdev', wavelength)[row, col, :]
    RI_stdev = np.tile(RI_stdev, (1, 1)) 

    Q = get_HARP2_variable(HARP2, 'q', wavelength)[row, col, :]
    Q = ((np.pi * (sun_earth_dist**2) * Q) / (pol_F0 * np.cos(np.deg2rad(SZA))))
    Q = np.tile(Q, (1, 1)) 
    Q_stdev = get_HARP2_variable(HARP2, 'q_stdev', wavelength)[row, col, :]

    U = get_HARP2_variable(HARP2, 'u', wavelength)[row, col, :]
    U = ((np.pi * (sun_earth_dist**2) * U) / (pol_F0 * np.cos(np.deg2rad(SZA))))
    U = np.tile(U, (1, 1)) 
    U_stdev = get_HARP2_variable(HARP2, 'u_stdev', wavelength)[row, col, :]

    Rp = np.sqrt(U**2 + Q**2)
    Rp = np.tile(Rp, (1, 1)) 
    Rp_stdev = np.sqrt(U_stdev**2 + Q_stdev**2)
    Rp_stdev = np.tile(Rp_stdev, (1, 1)) 

    DoLP = get_HARP2_variable(HARP2, 'dolp', wavelength)[row, col, :] # does this also need to be transformed?
    DoLP = np.tile(DoLP, (1, 1)) 
    DoLP_stdev = U_stdev = get_HARP2_variable(HARP2, 'dolp_stdev', wavelength)[row, col, :]
    DoLP_stdev = np.tile(DoLP_stdev, (1, 1)) 

    VZA = get_HARP2_variable(HARP2, 'sensor_zenith_angle', wavelength)[row, col, :]
    #print(VZA.shape)
    ## catches error when the array is all nans
    # ##! not sure that this should be negative
    # try:
    #     min_VZA_idx = np.nanargmin(VZA)
    #     VZA[0:min_VZA_idx] = -VZA[0:min_VZA_idx]  #! check
    #     # VZA[min_VZA_idx::] = -VZA[min_VZA_idx::]  #! check
    # except Exception as error:
    #     VZA = VZA
    
    ## altitude = get_HARP2_variable(HARP2, 'altitude', wavelength)[row, col] #! not in dataset
    longitude = get_HARP2_variable(HARP2, 'longitude', wavelength)[row, col]
    latitude = get_HARP2_variable(HARP2, 'latitude', wavelength)[row, col]

    data = xr.Dataset(
        {
             "RI": (["Wavelength", "VZA"], RI, {'long_name':'Total reflectance'}),
             "RI_stdev": (["Wavelength", "VZA"], RI_stdev),
             "Rp": (["Wavelength", "VZA"], Rp,  {'long_name':'Polarized reflectance'}),
             "Rp_stdev": (["Wavelength", "VZA"], Rp_stdev),                   
             "Q": (["Wavelength", "VZA"], Q),
             "U": (["Wavelength", "VZA"], U),
             "DoLP": (["Wavelength", "VZA"], DoLP),
             "DoLP_stdev": (["Wavelength", "VZA"], DoLP_stdev),
             'mean_SZA': ([], mean_SZA, {'long_name':'Mean solar zenith angle', 'units':'degrees'}),
             'mean_RAZ': ([], mean_RAZ,{'long_name':'Mean relative azimuth angle', 'units':'degrees'}), 
             'mean_latitude': ([], latitude, {'long_name':'Mean latitude'}), 
             'mean_longitude': ([], longitude, {'long_name':'Mean longitude'}), 
            #  'mean_altitude': ([], altitude, {'long_name':'Mean altitude'}),      
         },
         coords={
             "Wavelength": ("Wavelength", np.array([wavelength]), {'long_name':'RSP wavelengths', 'units':'nm'}),
             "VZA": ("VZA", VZA, {'long_name':'Viewing zenith angle', 'units':'degrees'}),
             "SZA": ("VZA", SZA, {'long_name':'Solar zenith angle', 'units':'degrees'}),
             "RAZ": ("VZA", RAZ, {'long_name':'Relative azimuth angle', 'units':'degrees'}),
             "SCATANG": ("VZA", scatang, {'long_name':'Scattering angle', 'units':'degrees'}),
        },
        attrs={'creation_date':datetime.now(),
            }
        
        )
    # data = data.dropna(dim='VZA')
    return data

def main():
    data_dir = config.data_dir
    path = '/Users/olivia/Downloads/code/snow_code/data/PACE/'
    HARP2_file ='PACE_HARP2_SIM.20220321T183042.L1C.5km.V03.nc'
    data = readHARP2(path+HARP2_file, 867, 9, 4).sel(Wavelength=867)
    return 


if __name__ == "__main__":
    ''' Purpose: Testing this python file. Code below does not run unless this file is executed as "main"
    '''
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    grandparent = os.path.dirname(parent)
    sys.path.append(grandparent)
    from inversion_config import inversion_config_oil as config
    cProfile.run('main()', sort='tottime')
    # data_dir = config.data_dir
    # path = '/Users/olivia/Downloads/code/snow_code/data/PACE/'
    # HARP2_file ='GOM_sub_PACE_HARP2_SIM.20220321T183042.L1C.5km.V03.h5'
    # import time
    # st = time.time()
    # data = readHARP2(path+HARP2_file, 867, 9, 4).sel(Wavelength=867)
    # print(time.time() - st)
    # print(data)
    # plt.plot(data.SCATANG, data.RI)
    # # plt.show()
    # print(f'VZA: {data.VZA.values}')
    # print(f'RAZ: {data.RAZ.values}')
    # print(f'SZA: {data.SZA.values}')

    # measurement_file_path = config.data_dir + config.measurement_file_name
    # measurement_file_path = config.data_dir + '/PACE/small_GOM_sub_PACE_HARP2.20220321T183000.L1C.5.2KM.V02.h5' 
    # create_HARP2_RGB(config.measurement_file_path)
    # plt.show()

    # HARP2_file = 'small_GOM_sub_PACE_HARP2.20220321T183000.L1C.5.2KM.V02.h5'
    # HARP2 = h5py.File(f'{path}/{HARP2_file}', "r")
    # wavelength = 867

    # SZA  = get_HARP2_variable(HARP2, 'solar_zenith', wavelength)[:]
    # mean_SZA = np.nanmean(SZA)
    # scatang = get_HARP2_variable(HARP2, 'scattering_angle', wavelength)[:]
    # VAA = get_HARP2_variable(HARP2, 'sensor_azimuth', wavelength)[:]
    # SAA = get_HARP2_variable(HARP2, 'solar_azimuth', wavelength)[:]
    # RAZ = SAA - VAA #! RAZ correct?
    # # RAZ = get_raz(SAA, VAA)
    # mean_RAZ = np.nanmean(RAZ)
    # VZA = get_HARP2_variable(HARP2, 'sensor_zenith', wavelength)[:]
    #     # catches error when the array is all nans
    # intensity_F0 = get_HARP2_variable(HARP2, 'intensity_F0', wavelength)[:]
    # intensity_F0_full = np.full_like(VZA, np.nan)
    # for i in range(intensity_F0_full.shape[1]):
    #     for j in range(intensity_F0_full.shape[2]):
    #         intensity_F0_full[:, i, j] = intensity_F0
    # pol_F0 = get_HARP2_variable(HARP2, 'polarization_F0', wavelength)[:]
    # # sun_earth_dist = HARP2.attrs['sun_earth_distance']
    # # sun_earth_dist  = 1.50*10**11 # r_0 (mean earth sun distance 1.5 10^11 m)
    # sun_earth_dist = 0.990849042172323 # AU
    # RI = get_HARP2_variable(HARP2, 'I', wavelength)[:]
    # RI = ((np.pi * (sun_earth_dist**2) * RI) / (intensity_F0_full * np.cos(np.deg2rad(SZA))))

    # fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18, 5))
    # fig.subplots_adjust(wspace=0.2, hspace=0.5)

    # vza = axs[0].pcolormesh(VZA[5])
    # sza = axs[1].pcolormesh(SZA[5])
    # raz = axs[2].pcolormesh(RAZ[5])
    # diff = axs[3].pcolormesh(RAZ[-1]-RAZ[0])
    # plt.colorbar(vza, ax=axs[0], )#ticks=np.arange(np.nanmin(RAZ[5]), np.nanmax(RAZ[5]) + 1, 0.1))
    # plt.colorbar(sza, ax=axs[1], )#ticks=np.arange(np.nanmin(SZA[5]), np.nanmax(SZA[5]) + 1, 0.1))
    # plt.colorbar(raz, ax=axs[2], )#ticks=np.arange(np.nanmin(RAZ[-1]-RAZ[0]), np.nanmax(RAZ[-1]-RAZ[0])))
    # plt.colorbar(diff, ax=axs[3], )#ticks=np.arange(np.nanmin(get_raz(SAA, VAA)[5]), np.nanmax(get_raz(SAA, VAA)[5]) + 1, 0.1))

    # axs[0].set_ylabel('along track pixel')
    # for i in range(len(axs)):
    #     axs[i].set_xlabel('across track pixel')
    # axs[0].set_title('iewing Zenith at Angle 5')
    # axs[1].set_title('Solar Zenith at angle 5')
    # axs[2].set_title('Relative Azimuth at Angle 5')
    # axs[3].set_title('Min - Max Relative Azimuth')
    # plt.show()

