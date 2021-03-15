def streamflow_performance_func(Q_obs, Q_sim, P_monthly):
    
    # Q_sim consists of 12 timeseries that need to be optimized for
    # Q_obs also consists of a timeseries that need to be optimized for simultaneously
    # This can be done in the spotpy function
    # Function 1-4 are purely based on the streamflow functions
    # function 5 also needs a timeseries of precipitation input 
    # That input needs to be called before using this function (and whill thus be used as input)
    
    
    # -1 Input timeseries need to be for the whole simulation period
    # Here this is divided into a spin-up, calibration and validation period
    # observation are available from only 2000-01-01 but simulations from 1997-01-01.
    # This is fixed with this preparation
    
    # spin-up period
    #start_spinup = 0
    #end_spinup = 1095
    
    # calibration period
    start_cal_obs = 0
    end_cal_obs = 1461
    
    start_cal_sim = 1095
    end_cal_sim = 2556
        
    start_cal_P = 0
    end_cal_P = 12*4
    
    # validation period
    start_val_obs = 1461
    end_val_obs = 2616
    
    start_val_sim = 2556
    end_val_sim = 3711
        
    start_val_P = 12*4
    end_val_P = 12*7 + 2
    
    # 0. import required packages and prepare flow arrays 
    import numpy as np
    import pandas as pd
    import hydrostats.metrics as hm
    from statsmodels.tsa.stattools import acf
    from scipy.spatial import distance

    # Prepare flow arrays calibration period
    Q_obs_cal = Q_obs[start_cal_obs:end_cal_obs]
    Q_sim_cal = Q_sim[start_cal_sim:end_cal_sim]
    P_monthly_cal = P_monthly[start_cal_P:end_cal_P]
    
    # Prepare flow arrays validation period
    Q_obs_val = Q_obs[start_val_obs:end_val_obs]
    Q_sim_val = Q_sim[start_val_sim:end_val_sim]
    P_monthly_val = P_monthly[start_val_P:end_val_P]
    
    
    # 1. KGE of the flow (assess high flows)
    KGE_cal = hm.kge_2012(Q_sim_cal, Q_obs_cal, return_all=False)
    KGE_val = hm.kge_2012(Q_sim_val, Q_obs_val, return_all=False)    

    
    # 2. KGE of the Box-Cox transformed flows (assess low flows)
    def BoxCox(Q_obs, Q_sim, lamda=0.25):
        Qm_obs = np.nanmean(Q_obs)
        Q_BC_obs = (Q_obs ** lamda - (0.01 * Qm_obs) ** lamda) / lamda
        Q_BC_sim = (Q_sim ** lamda - (0.01 * Qm_obs) ** lamda) / lamda
        return Q_BC_obs, Q_BC_sim
    
    Q_BC_obs_cal, Q_BC_sim_cal = BoxCox(Q_obs=Q_obs_cal, Q_sim=Q_sim_cal)
    Q_BC_obs_val, Q_BC_sim_val = BoxCox(Q_obs=Q_obs_val, Q_sim=Q_sim_val)
    
    KGE_BC_cal = hm.kge_2012(Q_BC_sim_cal, Q_BC_obs_cal, return_all=False)
    KGE_BC_val = hm.kge_2012(Q_BC_sim_val, Q_BC_obs_val, return_all=False)
    
    
    # 3. flow duration curve (assess distribution of flow values, use Box-Cox transformed values)
    # Sort arrays of observed and simulated timeseries and create mask for nan-values
    Q_BC_obs_cal_sort = np.sort(Q_BC_obs_cal)[::-1]
    Q_BC_sim_cal_sort = np.sort(Q_BC_sim_cal)[::-1]
    mask_cal = ~np.isnan(Q_BC_obs_cal_sort) & ~np.isnan(Q_BC_sim_cal_sort)
    
    Q_BC_obs_val_sort = np.sort(Q_BC_obs_val)[::-1]
    Q_BC_sim_val_sort = np.sort(Q_BC_sim_val)[::-1]
    mask_val = ~np.isnan(Q_BC_obs_val_sort) & ~np.isnan(Q_BC_sim_val_sort)   

    # calculate exceedence
    exceedence_obs_cal = (np.arange(1,len(Q_BC_obs_cal_sort)+1) / len(Q_BC_obs_cal_sort)) * 100
    exceedence_sim_cal = (np.arange(1,len(Q_BC_sim_cal_sort)+1) / len(Q_BC_sim_cal_sort)) * 100

    exceedence_obs_val = (np.arange(1,len(Q_BC_obs_val_sort)+1) / len(Q_BC_obs_val_sort)) * 100
    exceedence_sim_val = (np.arange(1,len(Q_BC_sim_val_sort)+1) / len(Q_BC_sim_val_sort)) * 100
    
    # assess performance based on KGE
    KGE_FDC_cal = hm.kge_2012(Q_BC_sim_cal_sort[mask_cal], Q_BC_obs_cal_sort[mask_cal], return_all=False)
    KGE_FDC_val = hm.kge_2012(Q_BC_sim_val_sort[mask_val], Q_BC_obs_val_sort[mask_val], return_all=False)

    
    # 4. autocorrelation function
    # mask no data values and calculate autocorrelation timeseries for 180 days
    mask_cal = ~np.isnan(Q_obs_cal) & ~np.isnan(Q_sim_cal)
    Q_obs_cal_acf = acf(x=Q_obs_cal[mask_cal], nlags=60, fft=True)
    Q_sim_cal_acf = acf(x=Q_sim_cal[mask_cal], nlags=60, fft=True)

    mask_val = ~np.isnan(Q_obs_val) & ~np.isnan(Q_sim_val)
    Q_obs_val_acf = acf(x=Q_obs_val[mask_val], nlags=60, fft=True)
    Q_sim_val_acf = acf(x=Q_sim_val[mask_val], nlags=60, fft=True)

    # asses performance based on KGE
    KGE_ACF_cal = hm.kge_2012(Q_sim_cal_acf, Q_obs_cal_acf, return_all=False)
    KGE_ACF_val = hm.kge_2012(Q_sim_val_acf, Q_obs_val_acf, return_all=False)
    
    
    # 5. monthly runoff coefficients

    ### Flow part
    # Calibration period
    # Set observed and simlated flow in dataframe
    data_cal = pd.DataFrame(data=Q_obs_cal, columns=['Q_obs']) 
    data_cal['Q_sim'] = Q_sim_cal

    # make daterange for full period and turn it into the index of the dataframe
    dates_cal = pd.date_range(start='2000-01-01', end='2003-12-31', freq='D')
    data_cal['date'] = dates_cal
    data_cal.set_index('date', inplace=True)
    
    # Validation period
    # Set observed and simlated flow in dataframe
    data_val = pd.DataFrame(data=Q_obs_val, columns=['Q_obs']) 
    data_val['Q_sim'] = Q_sim_val

    # make daterange for full period and turn it into the index of the dataframe
    dates_val = pd.date_range(start='2004-01-01', end='2007-02-28', freq='D')
    data_val['date'] = dates_val
    data_val.set_index('date', inplace=True) 

    # Resample the dataframe to monthly values and take the mean per day to exclude effect of nan values (nans are not counted)
    #Q_monthly = data.resample('M').mean()
    Q_monthly_cal = data_cal.resample('M').mean()
    Q_monthly_obs_cal = Q_monthly_cal['Q_obs']
    Q_monthly_sim_cal = Q_monthly_cal['Q_sim']
    
    Q_monthly_val = data_val.resample('M').mean()
    Q_monthly_obs_val = Q_monthly_val['Q_obs']
    Q_monthly_sim_val = Q_monthly_val['Q_sim']   
    
    # Convert from m^3/s to m^3/d (for equal comparison with precipitation values)
    Q_monthly_obs_cal = Q_monthly_obs_cal * 24 * 3600
    Q_monthly_sim_cal = Q_monthly_sim_cal * 24 * 3600
    
    Q_monthly_obs_val = Q_monthly_obs_val * 24 * 3600
    Q_monthly_sim_val = Q_monthly_sim_val * 24 * 3600

    
    ### Precipitation part
    # Already imported
    
    ### Runoff coefficient part
    # Calculate monthly runoff coefficients   
    Rc_obs_cal = np.array(Q_monthly_obs_cal) / np.array(P_monthly_cal)
    Rc_sim_cal = np.array(Q_monthly_sim_cal) / np.array(P_monthly_cal)
    
    Rc_obs_val = np.array(Q_monthly_obs_val) / np.array(P_monthly_val)
    Rc_sim_val = np.array(Q_monthly_sim_val) / np.array(P_monthly_val)    
    
    # Assess performance based on KGE
    KGE_Rc_cal = hm.kge_2012(Rc_sim_cal, Rc_obs_cal, return_all=False)
    KGE_Rc_val = hm.kge_2012(Rc_sim_val, Rc_obs_val, return_all=False)

    
    # 6. Perform mean calculation of all objective functions using weighted average
    weights = np.array([0.30, 0.25, 0.15, 0.15, 0.15])
    u_cal = np.array([KGE_cal, KGE_BC_cal, KGE_FDC_cal, KGE_ACF_cal, KGE_Rc_cal])
    mean_cal = np.average(u_cal, weights=weights)
    
    u_val = np.array([KGE_val, KGE_BC_val, KGE_FDC_val, KGE_ACF_val, KGE_Rc_val])
    mean_val = np.average(u_val, weights=weights)
    
    # Create results array
    # need some kind of index to be able to identify the results (mean KGE of cal stations)
    # The higher the mean KGE the better the model simulation!
    
    results_cal = np.array([mean_cal, u_cal[0], u_cal[1], u_cal[2], u_cal[3], u_cal[4]])
    results_val = np.array([mean_val, u_val[0], u_val[1], u_val[2], u_val[3], u_val[4]])
    
    return results_cal, results_val