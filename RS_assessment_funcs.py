### Write GRACE assessment in a script that assesses GRACE performance after saving of the TWSA netcdf file
# GRACE is quite fast (coarse spatial and temporal resolution) so can be done outside of the loop, also because of the rescaling (to monhtly) makes things very difficult inside the loop...

def TWSA_assessment_spatially(loc_TWSA_sim):
    ## IMPORT FUNCTIONS
    import xarray as xr
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error
    
    
    ## PREPARE DATA
    # Open files
    TWSA_sim_file = xr.open_dataset(loc_TWSA_sim)
    TWSA_obs_file = xr.open_dataset('TWSA_obs.nc')
    
    # Open TWSA data
    TWSA_sim_xr = TWSA_sim_file['TWSA']
    TWSA_obs_xr = TWSA_obs_file['TWSA']
    
    # turn into numpy arrays
    TWSA_sim_np = np.zeros(((62, 11, 9)))
    TWSA_obs_np = np.zeros(((62, 11, 9)))
    
    TWSA_sim_np[:, :, :] = TWSA_sim_xr[:, ::-1, :]
    TWSA_obs_np[:, :, :] = TWSA_obs_xr[:, ::-1, :]
    
    # reshape arrays 
    TWSA_sim_flat = np.reshape(TWSA_sim_np, (62, 99))
    TWSA_obs_flat = np.reshape(TWSA_obs_np, (62, 99))
    
    # create nan values in observations outside basin
    nan1_arr = np.where(np.isnan(TWSA_sim_flat), np.nan, 1)
    TWSA_obs_flat = TWSA_obs_flat * nan1_arr
    
    
    ## PERFORM Esp CALCULATION (this cannot be done in 2D and is also very fast like this!)
    Esp_TWSA_cal = np.zeros(62)
    Esp_TWSA_val = np.zeros(62)
    for i in range(62):
        
        if i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 17:
            Esp_TWSA_cal[i] = np.nan
            Esp_TWSA_val[i] = np.nan
            
        else: 
            # delete nan values
            TWSA_sim = TWSA_sim_flat[i][~np.isnan(TWSA_sim_flat[i])]                
            TWSA_obs = TWSA_obs_flat[i][~np.isnan(TWSA_obs_flat[i])]          

            # Validation cells are 3, 6, 7, 10, and 11 in count
            # Seperate calibration and validation cells
            TWSA_sim_val = np.array([TWSA_sim[3], TWSA_sim[6], TWSA_sim[10], TWSA_sim[11], TWSA_sim[16], TWSA_sim[17]])
            TWSA_obs_val = np.array([TWSA_obs[3], TWSA_obs[6], TWSA_obs[10], TWSA_obs[11], TWSA_obs[16], TWSA_obs[17]])

            TWSA_sim_cal = np.delete(TWSA_sim, np.array([17, 16, 11, 10, 6, 3]))
            TWSA_obs_cal = np.delete(TWSA_obs, np.array([17, 16, 11, 10, 6, 3]))
            
            
            # 1. Calculate spearman rank correlation of the flattened array of validation and calibration array
            rs_cal = spearmanr(a=TWSA_obs_cal, b=TWSA_sim_cal, axis=0, nan_policy='omit')[0]
            rs_val = spearmanr(a=TWSA_obs_val, b=TWSA_sim_val, axis=0, nan_policy='omit')[0]
    
            # 2. Calculate variability ratio
            # (assume means and std's are calculated wrt to space, not to time, because it is a spatial matching term)
            std_obs_cal = np.nanstd(TWSA_obs_cal)
            std_sim_cal = np.nanstd(TWSA_sim_cal)
            
            std_obs_val = np.nanstd(TWSA_obs_val)
            std_sim_val = np.nanstd(TWSA_sim_val)

            mean_obs_cal = np.nanmean(TWSA_obs_cal)
            mean_sim_cal = np.nanmean(TWSA_sim_cal)
            
            mean_obs_val = np.nanmean(TWSA_obs_val)
            mean_sim_val = np.nanmean(TWSA_sim_val)

            CV_obs_cal = std_obs_cal / mean_obs_cal
            CV_sim_cal = std_sim_cal / mean_sim_cal
            
            CV_obs_val = std_obs_val / mean_obs_val
            CV_sim_val = std_sim_val / mean_sim_val

            gamma_cal = CV_sim_cal / CV_obs_cal
            gamma_val = CV_sim_val / CV_obs_val
    
            # 3. spatial location matching term alpha 
            Z_obs_cal = (TWSA_obs_cal - mean_obs_cal) / std_obs_cal
            Z_sim_cal = (TWSA_sim_cal - mean_sim_cal) / std_sim_cal
            
            Z_obs_val = (TWSA_obs_val - mean_obs_val) / std_obs_val
            Z_sim_val = (TWSA_sim_val - mean_sim_val) / std_sim_val

            Z_obs_cal = Z_obs_cal[~np.isnan(Z_obs_cal)]
            Z_sim_cal = Z_sim_cal[~np.isnan(Z_sim_cal)]
            
            Z_obs_val = Z_obs_val[~np.isnan(Z_obs_val)]
            Z_sim_val = Z_sim_val[~np.isnan(Z_sim_val)]

            alpha_cal  = 1 - np.sqrt(mean_squared_error(y_true=Z_obs_cal, y_pred=Z_sim_cal))
            alpha_val  = 1 - np.sqrt(mean_squared_error(y_true=Z_obs_val, y_pred=Z_sim_val))

    
            # 4. Return Esp from its three components
            Esp_TWSA_cal[i] = 1 - np.sqrt((rs_cal - 1) ** 2 + (gamma_cal - 1) ** 2 + (alpha_cal - 1) ** 2)
            Esp_TWSA_val[i] = 1 - np.sqrt((rs_val - 1) ** 2 + (gamma_val - 1) ** 2 + (alpha_val - 1) ** 2)

        
        # Divide the TWSA dataset into two periods, one for calibration and one for evaluation
        # calibration period (2002 t/m feb 2005)   (32 months with data)
        # evaluation period  (mar 2005 - feb 2007) (24 months)
        # Take mean of arrays
        Esp_TWSA_mean_cal_cal = np.nanmean(Esp_TWSA_cal[0:38])     # calibration catchment, calibration period
        Esp_TWSA_mean_cal_val = np.nanmean(Esp_TWSA_cal[38:])      # calibration catchment, validation period
        Esp_TWSA_mean_val_cal = np.nanmean(Esp_TWSA_val[0:38])     # validation catchment, calibration period
        Esp_TWSA_mean_val_val = np.nanmean(Esp_TWSA_val[38:])      # validation catchment, validation period
        
    TWSA_sim_file.close()
    TWSA_obs_file.close()
        
    return Esp_TWSA_mean_cal_cal, Esp_TWSA_mean_cal_val, Esp_TWSA_mean_val_cal, Esp_TWSA_mean_val_val



## Here, also a function for GRACE assessment temporally is given. This may take much longer!

def TWSA_assessment_temporally(loc_TWSA_sim):
    ## IMPORT FUNCTIONS
    import xarray as xr
    import numpy as np
    
    # Get files and open datasets
    TWSA_obs_file = xr.open_dataset('TWSA_obs.nc')
    TWSA_sim_file = xr.open_dataset(loc_TWSA_sim)
    
    # Open TWSA dataset
    TWSA_obs_xr = TWSA_obs_file['TWSA']                   
    TWSA_sim_xr = TWSA_sim_file['TWSA']
    
    # Turn into numpy arrays
    TWSA_obs_np = np.zeros(((62, 11, 9)))
    TWSA_sim_np = np.zeros(((62, 11, 9)))
    
    TWSA_sim_np[:, :, :] = TWSA_sim_xr[:, ::-1, :]  
    TWSA_obs_np[:, :, :] = TWSA_obs_xr[:, ::-1, :]
    
    # Divide in calibration and validation period
    # calibration period
    TWSA_sim_np_cal = TWSA_sim_np[0:38, :, :]
    TWSA_obs_np_cal = TWSA_obs_np[0:38, :, :]

    # validation period
    TWSA_sim_np_val = TWSA_sim_np[38:, :, :]
    TWSA_obs_np_val = TWSA_obs_np[38:, :, :]
    
    
    ## Calculate KGE value in 3D!!!
    # Prepare means and standard deviations
    mean_TWSA_sim_cal = np.round(np.nanmean(TWSA_sim_np_cal, axis=0), decimals=2)
    mean_TWSA_sim_val = np.round(np.nanmean(TWSA_sim_np_val, axis=0), decimals=2)
    
    std_TWSA_sim_cal  = np.nanstd(TWSA_sim_np_cal, axis=0)
    std_TWSA_sim_val  = np.nanstd(TWSA_sim_np_val, axis=0)
    
    mean_TWSA_obs_cal = np.round(np.nanmean(TWSA_obs_np_cal, axis=0), decimals=2)
    mean_TWSA_obs_val = np.round(np.nanmean(TWSA_obs_np_val, axis=0), decimals=2)

    std_TWSA_obs_cal = np.nanstd(TWSA_obs_np_cal, axis=0)
    std_TWSA_obs_val = np.nanstd(TWSA_obs_np_val, axis=0)

    
    # 1. Calculate Pearson r correlation coefficients
    cov_TWSA_cal = np.nanmean((TWSA_sim_np_cal - mean_TWSA_sim_cal) * (TWSA_obs_np_cal - mean_TWSA_obs_cal), axis=0)
    cov_TWSA_val = np.nanmean((TWSA_sim_np_val - mean_TWSA_sim_val) * (TWSA_obs_np_val - mean_TWSA_obs_val), axis=0)

    Pearson_r_cal = cov_TWSA_cal / (std_TWSA_sim_cal * std_TWSA_obs_cal)
    Pearson_r_val = cov_TWSA_val / (std_TWSA_sim_val * std_TWSA_obs_val)
    
    # 2. bias ratio beta                                      # means are probably both zero so they are shifted with 5!!
    beta_cal = (mean_TWSA_sim_cal + 5) / (mean_TWSA_obs_cal + 5)
    beta_val = (mean_TWSA_sim_val + 5) / (mean_TWSA_obs_val + 5)  # everything + 5
    
    # 3. variability ratio gamma
    gamma_cal = (std_TWSA_sim_cal / (mean_TWSA_sim_cal + 5)) / (std_TWSA_obs_cal / (mean_TWSA_obs_cal + 5))
    gamma_val = (std_TWSA_sim_val / (mean_TWSA_sim_val + 5)) / (std_TWSA_obs_val / (mean_TWSA_obs_val + 5)) # every mean + 5
    
    # 4. KGE calculation
    KGE_cal = 1 - np.sqrt((Pearson_r_cal - 1) ** 2 + (beta_cal - 1) ** 2 + (gamma_cal - 1) ** 2)
    KGE_val = 1 - np.sqrt((Pearson_r_val - 1) ** 2 + (beta_val - 1) ** 2 + (gamma_val - 1) ** 2)
    
    # Divide area in calibration and validation catchtments
    mean_KGE_val_cal = np.nanmean([KGE_cal[2, 4], KGE_cal[3, 4], KGE_cal[4, 4], KGE_cal[5, 4], KGE_cal[4, 5], KGE_cal[5, 5]])
    mean_KGE_val_val = np.nanmean([KGE_val[2, 4], KGE_val[3, 4], KGE_val[4, 4], KGE_val[5, 4], KGE_val[4, 5], KGE_val[5, 5]])
    
    KGE_cal[2, 4], KGE_cal[3, 4], KGE_cal[4, 4], KGE_cal[5, 4], KGE_cal[4, 5], KGE_cal[5, 5] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    KGE_val[2, 4], KGE_val[3, 4], KGE_val[4, 4], KGE_val[5, 4], KGE_val[4, 5], KGE_val[5, 5] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    mean_KGE_cal_cal = np.nanmean(KGE_cal)
    mean_KGE_cal_val = np.nanmean(KGE_val)
    
    # mean_KGE_cal_cal = calibration catchment, calibration period
    # mean_KGE_cal_val = calibration catchment, validation period
    # mean_KGE_val_cal = validation catchment, calibration period
    # mean_KGE_val_val = validation catchment, validation period
    
    TWSA_sim_file.close()
    TWSA_obs_file.close()
    
    return mean_KGE_cal_cal, mean_KGE_cal_val, mean_KGE_val_cal, mean_KGE_val_val



def ET_assessment_spatially(ET_sim_dayi, NDVI_obs_dayi):
    ## IMPORT FUNCTIONS
    import xarray as xr
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error
    
    # Load calibration and validation catchments
    calibration_catch  = np.loadtxt('wflow_Volta_hbv2_0/staticmaps/calibration_catch.txt')  
    validation_catch   = np.loadtxt('wflow_Volta_hbv2_0/staticmaps/validation_catch.txt')    
    
    ## PREPARE DATA
    # Open files
    ET_sim_xr = ET_sim_dayi
    NDVI_obs_xr = NDVI_obs_dayi
    
    # make numpy arrays to store and modify the data
    ET_sim_np = np.zeros((220, 180))
    NDVI_obs_np = np.zeros((220, 180))
    
    # turn array around to be able to plot it right and insert the right values
    ET_sim_np[:, :] = ET_sim_xr[::-1, :]
    NDVI_obs_np[:, :] = NDVI_obs_xr
    
    # create nan values in observations outside basin
    nan1_arr = np.where(np.isnan(ET_sim_np), np.nan, 1)
    NDVI_obs_np = NDVI_obs_np * nan1_arr
    
    # Create nan values in simulation inside basin (Lake Volta)           # this works great!
    nan2_arr = np.where(np.isnan(NDVI_obs_np), np.nan, 1)
    ET_sim_np = ET_sim_np * nan2_arr
    
    # Introduce calibration and validation catchments
    ET_sim_np_cal = ET_sim_np * calibration_catch
    ET_sim_np_val = ET_sim_np * validation_catch
    
    NDVI_obs_np_cal = NDVI_obs_np * calibration_catch
    NDVI_obs_np_val = NDVI_obs_np * validation_catch
    
    # reshape arrays                                        
    ET_sim_cal_flat = np.reshape(ET_sim_np_cal, (39600))
    ET_sim_val_flat = np.reshape(ET_sim_np_val, (39600))

    NDVI_obs_cal_flat = np.reshape(NDVI_obs_np_cal, (39600))
    NDVI_obs_val_flat = np.reshape(NDVI_obs_np_val, (39600))
    
    # delete nan values
    ET_sim_cal = ET_sim_cal_flat[~np.isnan(ET_sim_cal_flat)]    
    ET_sim_val = ET_sim_val_flat[~np.isnan(ET_sim_val_flat)]              

    NDVI_obs_cal = NDVI_obs_cal_flat[~np.isnan(NDVI_obs_cal_flat)]
    NDVI_obs_val = NDVI_obs_val_flat[~np.isnan(NDVI_obs_val_flat)]
    
    
    ## PERFORM Esp CALCULATION in 1D (input is daily xr.dataset sim and obs)
    # 1. Calculate spearman rank correlation of the flattened array
    rs_cal = spearmanr(a=NDVI_obs_cal, b=ET_sim_cal, axis=0, nan_policy='omit')[0]
    rs_val = spearmanr(a=NDVI_obs_val, b=ET_sim_val, axis=0, nan_policy='omit')[0]

    # 2. Calculate variability ratio
    # (assume means and std's are calculated wrt to space, not to time, because it is a spatial matching term)
    std_obs_cal = np.nanstd(NDVI_obs_cal)
    std_obs_val = np.nanstd(NDVI_obs_val)

    std_sim_cal = np.nanstd(ET_sim_cal)
    std_sim_val = np.nanstd(ET_sim_val)

    mean_obs_cal = np.nanmean(NDVI_obs_cal)
    mean_obs_val = np.nanmean(NDVI_obs_val)

    mean_sim_cal = np.nanmean(ET_sim_cal)
    mean_sim_val = np.nanmean(ET_sim_val)

    CV_obs_cal = std_obs_cal / mean_obs_cal
    CV_obs_val = std_obs_val / mean_obs_val

    CV_sim_cal = std_sim_cal / mean_sim_cal
    CV_sim_val = std_sim_val / mean_sim_val

    gamma_cal = CV_sim_cal / CV_obs_cal
    gamma_val = CV_sim_val / CV_obs_val

    # 3. spatial location matching term alpha 
    Z_obs_cal = (NDVI_obs_cal - mean_obs_cal) / std_obs_cal
    Z_obs_val = (NDVI_obs_val - mean_obs_val) / std_obs_val

    Z_sim_cal = (ET_sim_cal - mean_sim_cal) / std_sim_cal
    Z_sim_val = (ET_sim_val - mean_sim_val) / std_sim_val

    alpha_cal  = 1 - np.sqrt(mean_squared_error(y_true=Z_obs_cal, y_pred=Z_sim_cal))
    alpha_val  = 1 - np.sqrt(mean_squared_error(y_true=Z_obs_val, y_pred=Z_sim_val))

    # Return Esp from its three components
    Esp_NDVI_ET_cal = 1 - np.sqrt((rs_cal - 1) ** 2 + (gamma_cal - 1) ** 2 + (alpha_cal - 1) ** 2)
    Esp_NDVI_ET_val = 1 - np.sqrt((rs_val - 1) ** 2 + (gamma_val - 1) ** 2 + (alpha_val - 1) ** 2)

    # Divide the ET/NDVI dataset into two periods, outside this function!
    return Esp_NDVI_ET_cal, Esp_NDVI_ET_val


def ET_assessment_temporally(loc_ET_sim):
    ## Import functions
    import xarray as xr
    import numpy as np
    
    # Load calibration and validation catchments
    calibration_catch  = np.loadtxt('wflow_Volta_hbv2_0/staticmaps/calibration_catch.txt')  
    validation_catch   = np.loadtxt('wflow_Volta_hbv2_0/staticmaps/validation_catch.txt')   
    
    ## Open xr datasets
    NDVI_obs = xr.open_dataset('NDVI_obs.nc')
    ET_sim = xr.open_dataset(loc_ET_sim)
    
    NDVI_obs_xr = NDVI_obs['NDVI']
    ET_sim_xr = ET_sim['ET']
    
    # Turn into numpy arrays
    NDVI_obs_np = np.zeros(((2616, 220, 180)))
    ET_sim_np = np.zeros(((2616, 220, 180)))
    
    ET_sim_np[:, :, :] = ET_sim_xr[:, ::-1, :]   # this may take quite some memory!
    NDVI_obs_np[:, :, :] = NDVI_obs_xr

    # calculate normalized ET_sim_np (in total or per ts?) now it is in total
    max_ET = np.nanmax(ET_sim_np)
    min_ET = np.nanmin(ET_sim_np)
    ET_sim_np_norm = (ET_sim_np - min_ET) / (max_ET - min_ET)
    
    # Divide the assessment in a calibration and validation period
    NDVI_obs_np_cal = NDVI_obs_np[0:1461, :, :]
    NDVI_obs_np_val = NDVI_obs_np[1461:, :, :]
    
    ET_sim_np_norm_cal = ET_sim_np_norm[0:1461, :, :]
    ET_sim_np_norm_val = ET_sim_np_norm[1461:, :, :]
    
    
    ## Calculate KGE value in 3D!!!
    # Prepare means and standard deviations
    mean_ET_sim_cal = np.nanmean(ET_sim_np_norm_cal, axis=0)
    mean_ET_sim_val = np.nanmean(ET_sim_np_norm_val, axis=0)

    std_ET_sim_cal  = np.nanstd(ET_sim_np_norm_cal, axis=0)
    std_ET_sim_val  = np.nanstd(ET_sim_np_norm_val, axis=0)

    mean_NDVI_obs_cal = np.nanmean(NDVI_obs_np_cal, axis=0)
    mean_NDVI_obs_val = np.nanmean(NDVI_obs_np_val, axis=0)

    std_NDVI_obs_cal = np.nanstd(NDVI_obs_np_cal, axis=0)
    std_NDVI_obs_val = np.nanstd(NDVI_obs_np_val, axis=0)

    
    # 1. Calculate Pearson r correlation coefficients
    cov_NDVI_ET_cal = np.nanmean((ET_sim_np_norm_cal - mean_ET_sim_cal) * (NDVI_obs_np_cal - mean_NDVI_obs_cal), axis=0)
    cov_NDVI_ET_val = np.nanmean((ET_sim_np_norm_val - mean_ET_sim_val) * (NDVI_obs_np_val - mean_NDVI_obs_val), axis=0)

    Pearson_r_cal = cov_NDVI_ET_cal / (std_ET_sim_cal * std_NDVI_obs_cal)
    Pearson_r_val = cov_NDVI_ET_val / (std_ET_sim_val * std_NDVI_obs_val)

    # 2. bias ratio beta
    beta_cal = mean_ET_sim_cal / mean_NDVI_obs_cal
    beta_val = mean_ET_sim_val / mean_NDVI_obs_val
    
    # 3. variability ratio gamma
    gamma_cal = (std_ET_sim_cal / mean_ET_sim_cal) / (std_NDVI_obs_cal / mean_NDVI_obs_cal)
    gamma_val = (std_ET_sim_val / mean_ET_sim_val) / (std_NDVI_obs_val / mean_NDVI_obs_val)
    
    # 4. KGE calculation
    KGE_cal = 1 - np.sqrt((Pearson_r_cal - 1) ** 2 + (beta_cal - 1) ** 2 + (gamma_cal - 1) ** 2)  # w/o loop still 51 seconds
    KGE_val = 1 - np.sqrt((Pearson_r_val - 1) ** 2 + (beta_val - 1) ** 2 + (gamma_val - 1) ** 2)

    
    ## Calculate KGE and its components in calibration and validation periods and catchments
    KGE_cal_cal = KGE_cal * calibration_catch             # calibration period, calibration catchment
    KGE_cal_val = KGE_cal * validation_catch              # calibration period, validation catchment
    KGE_val_cal = KGE_val * calibration_catch             # validation period, calibration catchment
    KGE_val_val = KGE_val * validation_catch              # validation period, validation catchment
    
    Pearson_r_cal_cal = Pearson_r_cal * calibration_catch             
    Pearson_r_cal_val = Pearson_r_cal * validation_catch              
    Pearson_r_val_cal = Pearson_r_val * calibration_catch             
    Pearson_r_val_val = Pearson_r_val * validation_catch              
    
    beta_cal_cal = beta_cal * calibration_catch             
    beta_cal_val = beta_cal * validation_catch              
    beta_val_cal = beta_val * calibration_catch            
    beta_val_val = beta_val * validation_catch 
    
    gamma_cal_cal = gamma_cal * calibration_catch             
    gamma_cal_val = gamma_cal * validation_catch              
    gamma_val_cal = gamma_val * calibration_catch            
    gamma_val_val = gamma_val * validation_catch 
    
    # Calculate means of KGE and its components
    mean_KGE_cal_cal = np.nanmean(KGE_cal_cal)
    mean_KGE_cal_val = np.nanmean(KGE_cal_val)
    mean_KGE_val_cal = np.nanmean(KGE_val_cal)
    mean_KGE_val_val = np.nanmean(KGE_val_val)
    
    mean_Pearson_r_cal_cal = np.nanmean(Pearson_r_cal_cal)
    mean_Pearson_r_cal_val = np.nanmean(Pearson_r_cal_val)
    mean_Pearson_r_val_cal = np.nanmean(Pearson_r_val_cal)
    mean_Pearson_r_val_val = np.nanmean(Pearson_r_val_val)
    
    mean_beta_cal_cal = np.nanmean(beta_cal_cal)
    mean_beta_cal_val = np.nanmean(beta_cal_val)
    mean_beta_val_cal = np.nanmean(beta_val_cal)
    mean_beta_val_val = np.nanmean(beta_val_val) 
    
    mean_gamma_cal_cal = np.nanmean(gamma_cal_cal)
    mean_gamma_cal_val = np.nanmean(gamma_cal_val)
    mean_gamma_val_cal = np.nanmean(gamma_val_cal)
    mean_gamma_val_val = np.nanmean(gamma_val_val)
    
    # Close datasets
    NDVI_obs.close()
    ET_sim.close()

    # Return only KGE's for now, but more is possible of course
    return mean_KGE_cal_cal, mean_KGE_cal_val, mean_KGE_val_cal, mean_KGE_val_val

## function that takes as input the ICF and creates a numpy Su_max map as output

def ICF_transformer(ICF_nf, ICF_f):
    import numpy as np
    # Calculate the Su_max for every combination of subcatchment and landuse (non-forest and forest)
    offset = [599.19503772, 625.03274585, 562.259546, 522.95253363, 401.11391317, 509.64177338, 534.37225299, 475.73219414, 533.26170669, 558.97329238, 513.43244853, 497.85879029, 426.11175142]
    slope = [-41.78375641, -41.23973506, -36.63365806, -33.08527239, -24.1950782, -45.04986909, -44.78285965, -31.07722248, -37.7781033,  -32.37036803, -30.7590952,  -32.34913525, -25.17571431]
    
    Su_max_nf = np.zeros(13)
    Su_max_f = np.zeros(13)
    for i in range(13):
        Su_max_nf[i] = offset[i] + ICF_nf * slope[i]
        Su_max_f[i] = offset[i] + ICF_f * slope[i]
    
    return Su_max_nf, Su_max_f