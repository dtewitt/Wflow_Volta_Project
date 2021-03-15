'''
Optimization SPOTPY using DDS for a distributed hydrological hbv model
:author: Daan te Witt
'''
# Import Model packages and scripts
from grpc4bmi.bmi_client_docker import BmiClientDocker
from spotpy.parameter import Uniform
from Streamflow_assessment import streamflow_performance_func

# Import basic packages
import numpy as np
import pandas as pd
from random import random
from scipy.spatial import distance
import xarray as xr

# Import copy and time packages
import shutil
from datetime import datetime
from cftime import num2date

# import netcdf write code
from var_to_xarray import var_to_xarray_GRACE
from var_to_xarray import var_to_xarray_ET
from var_to_xarray import var_to_xarray_SM

# import RS dataset assessment functions
from RS_assessment_funcs import TWSA_assessment_spatially
from RS_assessment_funcs import TWSA_assessment_temporally
from RS_assessment_funcs import ET_assessment_spatially
from RS_assessment_funcs import ET_assessment_temporally
from RS_assessment_funcs import ICF_transformer


class spot_setup_DDS1(object):
    """
    An implementation of the wflow_Volta_hbv parameter space 
    
    """
    
    ### DEFINE PARAMETER SPACE
    # Here the parameter space of the 24 calibration parameters is defined. 
    # Above the code, the parameter name in the wflow_hbv model is given.
    # If multiple lines of code are gives, this means that the parameter is implemented in a distributed way.
    
    # ICF non-forest, forest
    p1 = Uniform(low=1.0, high=3.0) 
    p2 = Uniform(low=2.5, high=6.0)
    
    # CEVPF non-forest, forest                      # actually max is 1.75 in paper
    p3 = Uniform(low=1.0, high=1.2)
    p4 = Uniform(low=1.6, high=2.5)

    # FC wetland, non-forest (hillslope + plateau), forest (hillslope + plateau)
#     p5 = Uniform(low=120, high=250)
#     p6 = Uniform(low=200, high=400)
#     p7 = Uniform(low=500, high=1200)
    
    # Beta
    p8 = Uniform(low=2.5, high=3.5)
    
    # LP
    p9 = Uniform(low=0.3, high=0.7)
    
    # PERC
    p10 = Uniform(low=0.1, high=8.0)           # Confine parameter space here!
    
    # Cflux
    p11 = Uniform(low=0.5, high=1.75)
    
    # SUZ
    p12 = Uniform(low=10, high=90)              # Confine parameter space here!
    
    # K4, K0, KQuickFlow
    #p13 = Uniform(low=0.04, high=0.12)
    p14 = Uniform(low=0.10, high=0.38)
    p15 = Uniform(low=0.10, high=0.50)
    
    # N wetland, forest (hillslope + plateau), non-forest hillslope, non-forest plateau     
    p16 = Uniform(low=0.15, high=0.35)
    p17 = Uniform(low=0.15, high=0.35)
    p18 = Uniform(low=0.05, high=0.12)
    p19 = Uniform(low=0.10, high=0.25)
    
#     # N_River 4, 5, 6 (7 is Lower Volta and 1, 2, 3 are not classified as rivers)
#     p20 = Uniform(low=0.030, high=0.050)
#     p21 = Uniform(low=0.030, high=0.050)
#     p22 = Uniform(low=0.030, high=0.043)

    p20 = Uniform(low=0.030, high=0.050)       # for subcatchments 1, 6, 7, 10, 11, 12    Delete this part if streamorder can be used again
    p21 = Uniform(low=0.030, high=0.043)       # for subcatchments 2, 3, 4, 5, 8, 9
   
    ### STORE SIMUMALTION RESULTS
    # Create table to store results of the streamflow performance
    results = np.zeros(((2, 14, 13)))
    results = results.reshape(2, 182)
    path = str('Results_DDS.txt')
    np.savetxt(str(path), results)
    
    
    def __init__(self,obj_func=None):
        self.obj_func = obj_func

    def simulation(self, x):
        
        ### CREATE NEW TABLES TO EXTRACT PARAMETER VALUES FROM
        # Get parameters values back
        p1, p2, p3, p4, p8, p9, p10, p11, p12, p14, p15, p16, p17, p18, p19, p20, p21 = x[:]

        # store parameters in tables
        ICF = [[1, 2],
               ['<,13]', '<,13]'],
               ['<,3]', '<,3]'], 
               [p1, p2]]
        
        CEVPF = [[1, 2], 
                 ['<,13]', '<,13]'], 
                 ['<,3]', '<,3]'],
                 [p3, p4]]
        
#         FC = [['<,2]', 1, 2],
#               ['<,13]', '<,13]', '<,13]'], 
#               [1, '[2,>', '[2,>'],
#               [p5, p6, p7]]
        Su_max_nf, Su_max_f = ICF_transformer(p1, p2)
        FC = [[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
              [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13], 
              ['<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]'],
              [Su_max_nf[0], Su_max_f[0], Su_max_nf[1], Su_max_f[1],Su_max_nf[2], Su_max_f[2],Su_max_nf[3], Su_max_f[3], Su_max_nf[4], Su_max_f[4], Su_max_nf[5], Su_max_f[5],Su_max_nf[6], Su_max_f[6],Su_max_nf[7], Su_max_f[7], Su_max_nf[8], Su_max_f[8], Su_max_nf[9], Su_max_f[9],Su_max_nf[10], Su_max_f[10],Su_max_nf[11], Su_max_f[11], Su_max_nf[12], Su_max_f[12]]]
        
        BetaSeepage = [['<,2]'], 
                       ['<,13]'], 
                       ['<,3]'],
                       [p8]]
        
        LP = [['<,2]'], 
              ['<,13]'], 
              ['<,3]'],
              [p9]]
        
        PERC = [['<,2]'], 
                ['<,13]'], 
                ['<,3]'],
                [p10]]
        
        Cflux = [['<,2]', '<,2]', '<,2]'], 
                 ['<,13]', '<,13]', '<,13]'], 
                 [1, 2, 3],
                 [p11, 0, 0]]
        
        SUZ = [['<,2]'], 
               ['<,13]'],
               ['<,3]'], 
               [p12]]
        
#         K4 = [['<,2]'], 
#               ['<,13]'],
#               ['<,3]'], 
#               [p13]]
        
        K0 = [['<,2]'], 
              ['<,13]'], 
              ['<,3]'],
              [p14]]
        
        KQuickFlow = [['<,2]'],
                      ['<,13]'], 
                      ['<,3]'],
                      [p15]]
        
        N = [['<,2]', '2', '1', '1'],
             ['<,13]', '<,13]', '<,13]', '<,13]'],
             [1, '[2,>', 2, 3],
             [p16, p17, p18, p19]]
        
#         N_River = [['<,3]', 4, 5, 6, 7],                                                 # first column can be deleted if problem can be solved!        
#                    ['<,13]', '<,13]', '<,13]', '<,13]', '<,13]'],                        # can then also be transformed to 2 columns (streamorder and N-values)
#                    ['<,3]', '<,3]', '<,3]', '<,3]', '<,3]'], 
#                    [0.030, p20, p21, p22, 0.15]]
        
        N_River = [['<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]'],                                                     
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],                      
                   ['<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]'], 
                   [p20, p21, p21, p21, p21, p20, p20, p21, p21, p20, p20, p20, 0.15]]
        
        
        # make dataframes from tables and transpose them
        df_ICF         = pd.DataFrame(ICF).transpose()
        df_CEVPF       = pd.DataFrame(CEVPF).transpose()
        df_FC          = pd.DataFrame(FC).transpose()
        df_BetaSeepage = pd.DataFrame(BetaSeepage).transpose()
        df_LP          = pd.DataFrame(LP).transpose()
        df_PERC        = pd.DataFrame(PERC).transpose()
        df_Cflux       = pd.DataFrame(Cflux).transpose()
        df_SUZ         = pd.DataFrame(SUZ).transpose()
        #df_K4          = pd.DataFrame(K4).transpose()
        df_K0          = pd.DataFrame(K0).transpose()
        df_KQuickFlow  = pd.DataFrame(KQuickFlow).transpose()
        df_N           = pd.DataFrame(N).transpose()
        df_N_River     = pd.DataFrame(N_River).transpose()
        
        
        ### WRITE UNIQUE FOLDERS TO STORE OUTPUT
        # Create unique ID for output
        millitime = datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f')
        output_name = '/mnt/home/user41/Working_dir_Daan/Streamflow_Calibration_test/wflow_Volta_hbv2_' + str(millitime)
        shutil.copytree("/mnt/home/user41/Working_dir_Daan/wflow_Volta_hbv2_0", output_name) 
        output_folder = output_name[34:]
        
        # Store parameter set in folder (how does this work for the results section?)
        param_set = np.zeros(17)
        param_set[:] = p1, p2, p3, p4, p8, p9, p10, p11, p12, p14, p15, p16, p17, p18, p19, p20, p21
        np.savetxt(str(output_folder) + '/param_set.txt', param_set)

        ### DEFINE PATHS TO INTBL FILES AND STORE NEW TABLES
        # Make new filepaths
        new_beta       = str(output_folder) + '/intbl/BetaSeepage.tbl'
        new_CEVPF      = str(output_folder) + '/intbl/CEVPF.tbl'
        new_Cflux      = str(output_folder) + '/intbl/Cflux.tbl'
        new_FC         = str(output_folder) + '/intbl/FC.tbl'
        new_ICF        = str(output_folder) + '/intbl/ICF.tbl'
        new_K0         = str(output_folder) + '/intbl/K0.tbl'
        #new_K4         = str(output_folder) + '/intbl/K4.tbl'
        new_KQuickFlow = str(output_folder) + '/intbl/KQuickFlow.tbl'
        new_LP         = str(output_folder) + '/intbl/LP.tbl'
        new_N_River    = str(output_folder) + '/intbl/N_River.tbl'
        new_N          = str(output_folder) + '/intbl/N.tbl'
        new_PERC       = str(output_folder) + '/intbl/PERC.tbl'
        new_SUZ        = str(output_folder) + '/intbl/SUZ.tbl'
        
        # save dataframes as tbl's
        df_ICF.to_csv(new_ICF, sep=' ', index=False, header=False)
        df_CEVPF.to_csv(new_CEVPF, sep=' ', index=False, header=False)
        df_FC.to_csv(new_FC, sep=' ', index=False, header=False)
        df_BetaSeepage.to_csv(new_beta, sep=' ', index=False, header=False)
        df_LP.to_csv(new_LP, sep=' ', index=False, header=False)         
        df_PERC.to_csv(new_PERC, sep=' ', index=False, header=False)         
        df_Cflux.to_csv(new_Cflux, sep=' ', index=False, header=False)
        df_SUZ.to_csv(new_SUZ, sep=' ', index=False, header=False)
        #df_K4.to_csv(new_K4, sep=' ', index=False, header=False)
        df_K0.to_csv(new_K0, sep=' ', index=False, header=False)
        df_KQuickFlow.to_csv(new_KQuickFlow, sep=' ', index=False, header=False)
        df_N.to_csv(new_N, sep=' ', index=False, header=False)        
        df_N_River.to_csv(new_N_River, sep=' ', index=False, header=False)
        
        
        ### CREATE NEW MODEL
        # Get Model
        #from grpc4bmi.bmi_client_docker import BmiClientDocker
        wflow_Volta_hbv = BmiClientDocker(image='ewatercycle/wflow-grpc4bmi:latest', image_port=55555, 
                                        input_dir=str(output_name),
                                        output_dir="./output")
        
        
        # Initialize model
        wflow_Volta_hbv.initialize('wflow_hbv.ini')
        
        
        ## RUN MODEL
        # Run model for 10+ years and store the LowerZoneStorage, UpperZoneStorage, Soil Moisture and Evapotranspiration
        # GRACE simulations are saved from the start of 2002
        # ET and SM simulations are saved from the start of 2000
        
        # Make some empty and zero array to fill with results
        TWS, ET, SM = [], [], []
        Esp_ET_cal_cal, Esp_ET_cal_val = np.zeros(1461), np.zeros(1461)
        Esp_ET_val_cal, Esp_ET_val_val = np.zeros(1155), np.zeros(1155)
        
        # Open NDVI dataset
        NDVI_obs = xr.open_dataset('NDVI_obs.nc')
        NDVI_obs = NDVI_obs['NDVI']
        
        # Run model
        for i in range(3711):
            wflow_Volta_hbv.update()

            if wflow_Volta_hbv.get_current_time() >= + wflow_Volta_hbv.get_start_time() + ((365 * 5) + 1) * 86400:
                TWS.append(var_to_xarray_GRACE(wflow_Volta_hbv, variable1="LowerZoneStorage", variable2="UpperZoneStorage", variable3="SoilMoisture"))

            if wflow_Volta_hbv.get_current_time() >= + wflow_Volta_hbv.get_start_time() + 365 * 3 * 86400:
                ETi = var_to_xarray_ET(wflow_Volta_hbv, variable="ActEvap")
                ET.append(ETi)
                
                SMi = var_to_xarray_SM(wflow_Volta_hbv, variable="SoilMoisture")
                SM.append(SMi)
                
                if wflow_Volta_hbv.get_current_time() < wflow_Volta_hbv.get_start_time() + (365 * 3 * 86400) + (((365 * 4) + 1) * 86400):
                    if i == 310 + 1095 or i == 1237 or i == 1274 or i == 1323 or i == 1442 or i == 1468 or i == 1754 or i == 2106 or i == 2452 or i == 2453 or i == 2576 or i == 2632 or i == 2637 or i == 2641 or i == 2672 or i == 2680 or i == 2750 or i == 3134 or i == 3165 or i == 3192 or i == 3267 or i == 3322 or i == 3417:
                        Esp_ET_cal_cal[i-1095], Esp_ET_cal_val[i-1095] = np.nan, np.nan
                        #print(i, 'if statements works')
                    else:
#                         print(ETi)
#                         print(NDVI_obs[i-1095])
                        Esp_ET_cal_cal[i-1095], Esp_ET_cal_val[i-1095] = ET_assessment_spatially(ETi, NDVI_obs[i-1095])
                
                else:
                    Esp_ET_val_cal[i-1095-1461], Esp_ET_val_val[i-1095-1461] = ET_assessment_spatially(ETi, NDVI_obs[i-1095])
                

        # Make xarray concatenated datasets from the simulated data, resample the TWS to monthly means and fill months without observations with nans
        TWS = xr.concat(TWS, dim='time')
        TWS = TWS.resample(time='1M').mean()
        TWS[0, :, :] = np.zeros((11, 9)) * np.nan
        TWS[1, :, :] = np.zeros((11, 9)) * np.nan
        TWS[2, :, :] = np.zeros((11, 9)) * np.nan
        TWS[5, :, :] = np.zeros((11, 9)) * np.nan
        TWS[6, :, :] = np.zeros((11, 9)) * np.nan
        TWS[17, :, :] = np.zeros((11, 9)) * np.nan
        TWSA = TWS - TWS.mean(axis=0)
        ET = xr.concat(ET, dim='time')
        SM = xr.concat(SM, dim='time')

        # Save xarray datasets as netcdf's
        TWSA.to_netcdf(str(output_folder) + '/TWSA_sim.nc', encoding={'TWSA':{"zlib": True, "complevel": 4}})
        ET.to_netcdf(str(output_folder) + '/ET_sim.nc', encoding={'ET':{"zlib": True, "complevel": 4}})
        SM.to_netcdf(str(output_folder) + '/SM_sim.nc', encoding={'SM':{"zlib": True, "complevel": 4}})

        
        ### FINALIZE MODEL AND DELETE NOT NEEDED OUTPUT
        # clean container
        wflow_Volta_hbv.finalize()
        del wflow_Volta_hbv
        
        
        ## ASSESS PERFORMANCE TWSA 
        # Spatially
        loc_TWSA_sim = str(output_folder) + '/TWSA_sim.nc'
        Esp_TWSA_mean_cal_cal, Esp_TWSA_mean_cal_val, Esp_TWSA_mean_val_cal, Esp_TWSA_mean_val_val = TWSA_assessment_spatially(loc_TWSA_sim)
        
        # Save results TWSA in textfile in column 1
        RS_results = np.loadtxt(str(output_folder) + '/RS_results.txt')
        RS_results[0, 0], RS_results[1, 0], RS_results[2, 0], RS_results[3, 0] = Esp_TWSA_mean_cal_cal, Esp_TWSA_mean_cal_val, Esp_TWSA_mean_val_cal, Esp_TWSA_mean_val_val
        
        # Temporally
        KGE_TWSA_mean_cal_cal, KGE_TWSA_mean_cal_val, KGE_TWSA_mean_val_cal, KGE_TWSA_mean_val_val = TWSA_assessment_temporally(loc_TWSA_sim)
        RS_results[4, 0], RS_results[5, 0], RS_results[6, 0], RS_results[7, 0] = KGE_TWSA_mean_cal_cal, KGE_TWSA_mean_cal_val, KGE_TWSA_mean_val_cal, KGE_TWSA_mean_val_val
        
        
        ## ASSESS PERFORMANCE ET NDVI
        # Spatially
        Esp_ET_mean_cal_cal, Esp_ET_mean_cal_val = np.nanmean(Esp_ET_cal_cal), np.nanmean(Esp_ET_cal_val)
        Esp_ET_mean_val_cal, Esp_ET_mean_val_val = np.nanmean(Esp_ET_val_cal), np.nanmean(Esp_ET_val_val)
        RS_results[0, 1], RS_results[1, 1], RS_results[2, 1], RS_results[3, 1] = Esp_ET_mean_cal_cal, Esp_ET_mean_cal_val, Esp_ET_mean_val_cal, Esp_ET_mean_val_val
        
        # Temporally
        loc_ET_sim = str(output_folder) + '/ET_sim.nc'
        KGE_ET_mean_cal_cal, KGE_ET_mean_cal_val, KGE_ET_mean_val_cal, KGE_ET_mean_val_val = ET_assessment_temporally(loc_ET_sim)
        RS_results[4, 1], RS_results[5, 1], RS_results[6, 1], RS_results[7, 1] = KGE_ET_mean_cal_cal, KGE_ET_mean_cal_val, KGE_ET_mean_val_cal, KGE_ET_mean_val_val
        np.savetxt(str(output_folder) + '/RS_results.txt', RS_results)
        
    
        # get streamflow csv and put in the RS assessment results in the first columns which is not used!           # Still update this part
        sim = pd.read_csv(str(output_folder) + '/run_default/Runoff_gauges.csv', index_col='# Timestep') 
        sim = sim.iloc[:, 0:].to_numpy()
        sim[0, 0], sim[1, 0], sim[2, 0], sim[3, 0] = Esp_TWSA_mean_cal_cal, Esp_TWSA_mean_cal_val, Esp_TWSA_mean_val_cal, Esp_TWSA_mean_val_val
        sim[4, 0], sim[5, 0], sim[6, 0], sim[7, 0] = KGE_TWSA_mean_cal_cal, KGE_TWSA_mean_cal_val, KGE_TWSA_mean_val_cal, KGE_TWSA_mean_val_val
        sim[8, 0], sim[9, 0], sim[10, 0], sim[11, 0] = Esp_ET_mean_cal_cal, Esp_ET_mean_cal_val, Esp_ET_mean_val_cal, Esp_ET_mean_val_val
        sim[12, 0], sim[13, 0], sim[14, 0], sim[15, 0] = KGE_ET_mean_cal_cal, KGE_ET_mean_cal_val, KGE_ET_mean_val_cal, KGE_ET_mean_val_val        
        
    
        # Delete files that are not needed
        shutil.rmtree(str(output_name) + '/inmaps')
        shutil.rmtree(str(output_name) + '/instate')
        shutil.rmtree(str(output_name) + '/intbl')
        shutil.rmtree(str(output_name) + '/intss')
        shutil.rmtree(str(output_name) + '/staticmaps')
        
        # return streamflow
        return sim

        
    def evaluation(self):
        ### GET EVALUATION DATA OF THE MODEL
        eva = pd.read_csv('wflow_Volta_hbv2_0/intss/Streamflow_calibration_data.csv', delimiter=';', skiprows=0, index_col='Date')
        eva[eva==-9999.00] = np.nan
        eva = eva.iloc[:, 1:].to_numpy()
        return eva

    def objectivefunction(self, evaluation, simulation, params=None):
        ###
        # Get monthly rainfall data for runoff calculation coefficients
        P_Rc = pd.read_csv('wflow_Volta_hbv2_0/intss/monthly_P_arrays_per_subcatchment.csv', delimiter=',', index_col='Unnamed: 0')
        P_Rc = P_Rc.to_numpy()
        
        
        #SPOTPY expects to get one or multiple values back, that define the performence of the model run
        if not self.obj_func:
            # Return results of calibration and validation period (shape is mean_KGE(5), KGE, KGE_BC, KGE_FDC, KGE_ACF, KGE_Rc)
            simulation_Q = simulation[:, 1:]
            RS_results = simulation[0:24, 0]

            # Results calibration catchment BV
            res_cal1, res_val1 = streamflow_performance_func(evaluation[:, 0], simulation_Q[:, 0], P_Rc[0, :])
            res_cal2, res_val2 = streamflow_performance_func(evaluation[:, 1], simulation_Q[:, 1], P_Rc[1, :])
            res_cal3, res_val3 = streamflow_performance_func(evaluation[:, 2], simulation_Q[:, 2], P_Rc[2, :])
            res_cal4, res_val4 = streamflow_performance_func(evaluation[:, 3], simulation_Q[:, 3], P_Rc[3, :])
            res_cal5, res_val5 = streamflow_performance_func(evaluation[:, 4], simulation_Q[:, 4], P_Rc[4, :])
            
            # Results validaton catchment WV
            res_cal6, res_val6 = streamflow_performance_func(evaluation[:, 5], simulation_Q[:, 5], P_Rc[5, :])
            res_cal7, res_val7 = streamflow_performance_func(evaluation[:, 6], simulation_Q[:, 6], P_Rc[6, :])
            res_cal8, res_val8 = streamflow_performance_func(evaluation[:, 7], simulation_Q[:, 7], P_Rc[7, :])
            res_cal9, res_val9 = streamflow_performance_func(evaluation[:, 8], simulation_Q[:, 8], P_Rc[8, :])
            
            # Results calibration catchment Oti
            res_cal10, res_val10 = streamflow_performance_func(evaluation[:, 9], simulation_Q[:, 9], P_Rc[9, :])
            res_cal11, res_val11 = streamflow_performance_func(evaluation[:, 10], simulation_Q[:, 10], P_Rc[10, :])
            res_cal12, res_val12 = streamflow_performance_func(evaluation[:, 11], simulation_Q[:, 11], P_Rc[11, :])  
            
            
            # Calibration catchment calibration period
            mean_mean_cal    = np.mean(np.array([res_cal1[0], res_cal2[0], res_cal3[0], res_cal4[0], res_cal5[0], res_cal10[0], res_cal11[0], res_cal12[0]]))
            mean_KGE_cal     = np.mean(np.array([res_cal1[1], res_cal2[1], res_cal3[1], res_cal4[1], res_cal5[1], res_cal10[1], res_cal11[1], res_cal12[1]]))
            mean_KGE_BC_cal  = np.mean(np.array([res_cal1[2], res_cal2[2], res_cal3[2], res_cal4[2], res_cal5[2], res_cal10[2], res_cal11[2], res_cal12[2]]))
            mean_KGE_FDC_cal = np.mean(np.array([res_cal1[3], res_cal2[3], res_cal3[3], res_cal4[3], res_cal5[3], res_cal10[3], res_cal11[3], res_cal12[3]]))
            mean_KGE_ACF_cal = np.mean(np.array([res_cal1[4], res_cal2[4], res_cal3[4], res_cal4[4], res_cal5[4], res_cal10[4], res_cal11[4], res_cal12[4]]))
            mean_KGE_Rc_cal  = np.mean(np.array([res_cal1[5], res_cal2[5], res_cal3[5], res_cal4[5], res_cal5[5], res_cal10[5], res_cal11[5], res_cal12[5]]))
            
            # Calibration catchment validation period
            mean_mean_val    = np.mean(np.array([res_val1[0], res_val2[0], res_val3[0], res_val4[0], res_val5[0], res_val10[0], res_val11[0], res_val12[0]]))
            mean_KGE_val     = np.mean(np.array([res_val1[1], res_val2[1], res_val3[1], res_val4[1], res_val5[1], res_val10[1], res_val11[1], res_val12[1]]))
            mean_KGE_BC_val  = np.mean(np.array([res_val1[2], res_val2[2], res_val3[2], res_val4[2], res_val5[2], res_val10[2], res_val11[2], res_val12[2]]))
            mean_KGE_FDC_val = np.mean(np.array([res_val1[3], res_val2[3], res_val3[3], res_val4[3], res_val5[3], res_val10[3], res_val11[3], res_val12[3]]))
            mean_KGE_ACF_val = np.mean(np.array([res_val1[4], res_val2[4], res_val3[4], res_val4[4], res_val5[4], res_val10[4], res_val11[4], res_val12[4]]))
            mean_KGE_Rc_val  = np.mean(np.array([res_val1[5], res_val2[5], res_val3[5], res_val4[5], res_val5[5],  res_val10[5], res_val11[5], res_val12[5]]))
            
            # Validation catchment, calibration period
            mean_mean_cal2    = np.mean(np.array([res_cal6[0], res_cal7[0], res_cal8[0], res_cal9[0]]))
            mean_KGE_cal2     = np.mean(np.array([res_cal6[1], res_cal7[1], res_cal8[1], res_cal9[1]]))
            mean_KGE_BC_cal2  = np.mean(np.array([res_cal6[2], res_cal7[2], res_cal8[2], res_cal9[2]]))
            mean_KGE_FDC_cal2 = np.mean(np.array([res_cal6[3], res_cal7[3], res_cal8[3], res_cal9[3]]))
            mean_KGE_ACF_cal2 = np.mean(np.array([res_cal6[4], res_cal7[4], res_cal8[4], res_cal9[4]]))
            mean_KGE_Rc_cal2  = np.mean(np.array([res_cal6[5], res_cal7[5], res_cal8[5], res_cal9[5]]))
            
            # Validation catchment, validation period
            mean_mean_val2    = np.mean(np.array([res_val6[0], res_val7[0], res_val8[0], res_val9[0]]))
            mean_KGE_val2     = np.mean(np.array([res_val6[1], res_val7[1], res_val8[1], res_val9[1]]))
            mean_KGE_BC_val2  = np.mean(np.array([res_val6[2], res_val7[2], res_val8[2], res_val9[2]]))
            mean_KGE_FDC_val2 = np.mean(np.array([res_val6[3], res_val7[3], res_val8[3], res_val9[3]]))
            mean_KGE_ACF_val2 = np.mean(np.array([res_val6[4], res_val7[4], res_val8[4], res_val9[4]]))
            mean_KGE_Rc_val2  = np.mean(np.array([res_val6[5], res_val7[5], res_val8[5], res_val9[5]]))
            
            # Take mean of the performance functions for all streamflow stations
            u = np.array([res_cal1[0], res_cal2[0], res_cal3[0], res_cal4[0], res_cal5[0], res_cal10[0], res_cal11[0], res_cal12[0]]) # station 6, 7, 8 and 9 are used for validation
            mean_KGE_cal_stations = np.mean(u)
            
            # save results in array with shape (14, 13) (mean_KGE_cal_stations, cal results(6), val_results(6) for calibration and validation catchments)
            # create mean_cal and mean_val arrays
            means_cal = np.array([mean_mean_cal, mean_KGE_cal, mean_KGE_BC_cal, mean_KGE_FDC_cal, mean_KGE_ACF_cal, mean_KGE_Rc_cal])
            means_val = np.array([mean_mean_val, mean_KGE_val, mean_KGE_BC_val, mean_KGE_FDC_val, mean_KGE_ACF_val, mean_KGE_Rc_val])
            means_cal2 = np.array([mean_mean_cal2, mean_KGE_cal2, mean_KGE_BC_cal2, mean_KGE_FDC_cal2, mean_KGE_ACF_cal2, mean_KGE_Rc_cal2])
            means_val2 = np.array([mean_mean_val2, mean_KGE_val2, mean_KGE_BC_val2, mean_KGE_FDC_val2, mean_KGE_ACF_val2, mean_KGE_Rc_val2])
            
            # create the array to save
            data = np.zeros((1, 14, 13))
            data[0, :, 0] = mean_KGE_cal_stations
            data[0, 0, 1:7], data[0, 0, 7:] = res_cal1, res_val1
            data[0, 1, 1:7], data[0, 1, 7:] = res_cal2, res_val2
            data[0, 2, 1:7], data[0, 2, 7:] = res_cal3, res_val3
            data[0, 3, 1:7], data[0, 3, 7:] = res_cal4, res_val4
            data[0, 4, 1:7], data[0, 4, 7:] = res_cal5, res_val5
            data[0, 5, 1:7], data[0, 5, 7:] = res_cal6, res_val6
            data[0, 6, 1:7], data[0, 6, 7:] = res_cal7, res_val7
            data[0, 7, 1:7], data[0, 7, 7:] = res_cal8, res_val8
            data[0, 8, 1:7], data[0, 8, 7:] = res_cal9, res_val9
            data[0, 9, 1:7], data[0, 9, 7:] = res_cal10, res_val10
            data[0, 10, 1:7], data[0, 10, 7:] = res_cal11, res_val11
            data[0, 11, 1:7], data[0, 11, 7:] = res_cal12, res_val12
            data[0, 12, 1:7], data[0, 12, 7:] = means_cal, means_val
            data[0, 13, 1:7], data[0, 13, 7:] = means_cal2, means_val2
            
            # Save results, both in total and separate file per run
            results = np.array((np.loadtxt('Results_DDS.txt')))
            results = results.reshape(int(np.shape(results)[0]), 14, 13)
            results = np.append(results, data, axis=0)
            results = results.reshape(results.shape[0], 182)
            np.savetxt('Results_DDS.txt', results)
            
            # load TWSA results and caculate mean of mean Esp TWSA and mean of stations of KGE (which is a mean of 5 KGE's)
            mean_Esp_TWSA = RS_results[0]
            mean_Esp_ET   = RS_results[2]
            mean_Esp_SM   = RS_results[4]
            
            like = mean_KGE_cal_stations
            
        else:
            # THIS PART IS NOT USED
            #Way to ensure on flexible spot setup class
            ED_1 = streamflow_performance_func(evaluation[:, 0], simulation[:, 0], P_Rc[0, :])
            ED_2 = streamflow_performance_func(evaluation[:, 1], simulation[:, 1], P_Rc[1, :])
            ED_3 = streamflow_performance_func(evaluation[:, 2], simulation[:, 2], P_Rc[2, :])
            ED_4 = streamflow_performance_func(evaluation[:, 3], simulation[:, 3], P_Rc[3, :])
            ED_5 = streamflow_performance_func(evaluation[:, 4], simulation[:, 4], P_Rc[4, :])
            ED_6 = streamflow_performance_func(evaluation[:, 5], simulation[:, 5], P_Rc[5, :])
            ED_7 = streamflow_performance_func(evaluation[:, 6], simulation[:, 6], P_Rc[6, :])
            ED_8 = streamflow_performance_func(evaluation[:, 7], simulation[:, 7], P_Rc[7, :])
            ED_9 = streamflow_performance_func(evaluation[:, 8], simulation[:, 8], P_Rc[8, :])
            ED_10 = streamflow_performance_func(evaluation[:, 9], simulation[:, 9], P_Rc[9, :])
            ED_11 = streamflow_performance_func(evaluation[:, 10], simulation[:, 10], P_Rc[10, :])
            ED_12 = streamflow_performance_func(evaluation[:, 11], simulation[:, 11], P_Rc[11, :])       
            
            # Take ED of the performance functions for all streamflow stations
            u = np.array([ED_1[0], ED_2[0], ED_3[0], ED_4[0], ED_5[0], ED_6[0], ED_7[0], ED_8[0], ED_9[0], ED_10[0], ED_11[0], ED_12[0]])
            v = np.ones(12)
            like = -distance.euclidean(u, v)
            
        return like

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     '''
# Optimization SPOTPY using DDS for a distributed hydrological hbv model
# :author: Daan te Witt
# '''
# # Import Model packages and scripts
# from grpc4bmi.bmi_client_docker import BmiClientDocker
# from spotpy.parameter import Uniform
# from Streamflow_assessment import streamflow_performance_func

# # Import basic packages
# import numpy as np
# import pandas as pd
# from random import random
# from scipy.spatial import distance
# import xarray as xr

# # Import copy and time packages
# import shutil
# from datetime import datetime
# from cftime import num2date

# # import netcdf write code
# from var_to_xarray import var_to_xarray_GRACE
# from var_to_xarray import var_to_xarray_ET
# from var_to_xarray import var_to_xarray_SM

# # import RS dataset assessment functions
# from RS_assessment_funcs import TWSA_assessment


class spot_setup_DDS2(object):
    """
    An implementation of the wflow_Volta_hbv parameter space 
    
    """
    
    ### DEFINE PARAMETER SPACE
    # Here the parameter space of the 24 calibration parameters is defined. 
    # Above the code, the parameter name in the wflow_hbv model is given.
    # If multiple lines of code are gives, this means that the parameter is implemented in a distributed way.
    
    # ICF non-forest, forest
    p1 = Uniform(low=1.0, high=3.0) 
    p2 = Uniform(low=2.5, high=6.0)
    
    # CEVPF non-forest, forest                      # actually max is 1.75 in paper
    p3 = Uniform(low=1.0, high=1.2)
    p4 = Uniform(low=1.6, high=2.5)

    # FC wetland, non-forest (hillslope + plateau), forest (hillslope + plateau)
    p5 = Uniform(low=120, high=250)
    p6 = Uniform(low=200, high=400)
    p7 = Uniform(low=500, high=1200)
    
    # Beta
    p8 = Uniform(low=2.5, high=3.5)
    
    # LP
    p9 = Uniform(low=0.3, high=0.7)
    
    # PERC
    p10 = Uniform(low=0.1, high=8.0)           # Confine parameter space here!
    
    # Cflux
    p11 = Uniform(low=0.5, high=1.75)
    
    # SUZ
    p12 = Uniform(low=10, high=90)              # Confine parameter space here!
    
    # K4, K0, KQuickFlow
    p13 = Uniform(low=0.04, high=0.12)
    p14 = Uniform(low=0.10, high=0.38)
    p15 = Uniform(low=0.10, high=0.50)
    
    # N wetland, forest (hillslope + plateau), non-forest hillslope, non-forest plateau     
    p16 = Uniform(low=0.15, high=0.35)
    p17 = Uniform(low=0.15, high=0.35)
    p18 = Uniform(low=0.05, high=0.12)
    p19 = Uniform(low=0.10, high=0.25)
    
    # N_River 4, 5, 6 (7 is Lower Volta and 1, 2, 3 are not classified as rivers)
#     p20 = Uniform(low=0.030, high=0.050)
#     p21 = Uniform(low=0.030, high=0.050)
#     p22 = Uniform(low=0.030, high=0.043)

    p20 = Uniform(low=0.030, high=0.050)       # for subcatchments 1, 6, 7, 10, 11, 12    Delete this part if streamorder can be used again
    p21 = Uniform(low=0.030, high=0.043)       # for subcatchments 2, 3, 4, 5, 8, 9
   
    ### STORE SIMUMALTION RESULTS
    # Create table to store results of the streamflow performance
    results = np.zeros(((2, 14, 13)))
    results = results.reshape(2, 182)
    path = str('Results_DDS.txt')
    np.savetxt(str(path), results)
    
    
    def __init__(self,obj_func=None):
        self.obj_func = obj_func

    def simulation(self, x):
        
        ### CREATE NEW TABLES TO EXTRACT PARAMETER VALUES FROM
        # Get parameters values back
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21 = x[:]

        # store parameters in tables
        ICF = [[1, 2],
               ['<,13]', '<,13]'],
               ['<,3]', '<,3]'], 
               [p1, p2]]
        
        CEVPF = [[1, 2], 
                 ['<,13]', '<,13]'], 
                 ['<,3]', '<,3]'],
                 [p3, p4]]
        
        FC = [['<,2]', 1, 2],
              ['<,13]', '<,13]', '<,13]'], 
              [1, '[2,>', '[2,>'],
              [p5, p6, p7]]
        
        BetaSeepage = [['<,2]'], 
                       ['<,13]'], 
                       ['<,3]'],
                       [p8]]
        
        LP = [['<,2]'], 
              ['<,13]'], 
              ['<,3]'],
              [p9]]
        
        PERC = [['<,2]'], 
                ['<,13]'], 
                ['<,3]'],
                [p10]]
        
        Cflux = [['<,2]', '<,2]', '<,2]'], 
                 ['<,13]', '<,13]', '<,13]'], 
                 [1, 2, 3],
                 [p11, 0, 0]]
        
        SUZ = [['<,2]'], 
               ['<,13]'],
               ['<,3]'], 
               [p12]]
        
        K4 = [['<,2]'], 
              ['<,13]'],
              ['<,3]'], 
              [p13]]
        
        K0 = [['<,2]'], 
              ['<,13]'], 
              ['<,3]'],
              [p14]]
        
        KQuickFlow = [['<,2]'],
                      ['<,13]'], 
                      ['<,3]'],
                      [p15]]
        
        N = [['<,2]', '2', '1', '1'],
             ['<,13]', '<,13]', '<,13]', '<,13]'],
             [1, '[2,>', 2, 3],
             [p16, p17, p18, p19]]
        
#         N_River = [['<,3]', 4, 5, 6, 7],                                                 # first column can be deleted if problem can be solved!        
#                    ['<,13]', '<,13]', '<,13]', '<,13]', '<,13]'],                        # can then also be transformed to 2 columns (streamorder and N-values)
#                    ['<,3]', '<,3]', '<,3]', '<,3]', '<,3]'], 
#                    [0.030, p20, p21, p22, 0.15]]
        
        N_River = [['<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]', '<,2]'],                                                     
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],                      
                   ['<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]', '<,3]'], 
                   [p20, p21, p21, p21, p21, p20, p20, p21, p21, p20, p20, p20, 0.15]]
        
        # make dataframes from tables and transpose them
        df_ICF         = pd.DataFrame(ICF).transpose()
        df_CEVPF       = pd.DataFrame(CEVPF).transpose()
        df_FC          = pd.DataFrame(FC).transpose()
        df_BetaSeepage = pd.DataFrame(BetaSeepage).transpose()
        df_LP          = pd.DataFrame(LP).transpose()
        df_PERC        = pd.DataFrame(PERC).transpose()
        df_Cflux       = pd.DataFrame(Cflux).transpose()
        df_SUZ         = pd.DataFrame(SUZ).transpose()
        df_K4          = pd.DataFrame(K4).transpose()
        df_K0          = pd.DataFrame(K0).transpose()
        df_KQuickFlow  = pd.DataFrame(KQuickFlow).transpose()
        df_N           = pd.DataFrame(N).transpose()
        df_N_River     = pd.DataFrame(N_River).transpose()
        
        
        ### WRITE UNIQUE FOLDERS TO STORE OUTPUT
        # Create unique ID for output
        millitime = datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f')
        output_name = '/mnt/home/user41/Working_dir_Daan/GRACE_Calibration_test/wflow_Volta_hbv2_' + str(millitime)
        shutil.copytree("/mnt/home/user41/Working_dir_Daan/wflow_Volta_hbv2_0", output_name) 
        output_folder = output_name[34:]
        
        # Store parameter set in folder (how does this work for the results section?)
        param_set = np.zeros(21)
        param_set[:] = p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21
        np.savetxt(str(output_folder) + '/param_set.txt', param_set)

        ### DEFINE PATHS TO INTBL FILES AND STORE NEW TABLES
        # Make new filepaths
        new_beta       = str(output_folder) + '/intbl/BetaSeepage.tbl'
        new_CEVPF      = str(output_folder) + '/intbl/CEVPF.tbl'
        new_Cflux      = str(output_folder) + '/intbl/Cflux.tbl'
        new_FC         = str(output_folder) + '/intbl/FC.tbl'
        new_ICF        = str(output_folder) + '/intbl/ICF.tbl'
        new_K0         = str(output_folder) + '/intbl/K0.tbl'
        new_K4         = str(output_folder) + '/intbl/K4.tbl'
        new_KQuickFlow = str(output_folder) + '/intbl/KQuickFlow.tbl'
        new_LP         = str(output_folder) + '/intbl/LP.tbl'
        new_N_River    = str(output_folder) + '/intbl/N_River.tbl'
        new_N          = str(output_folder) + '/intbl/N.tbl'
        new_PERC       = str(output_folder) + '/intbl/PERC.tbl'
        new_SUZ        = str(output_folder) + '/intbl/SUZ.tbl'
        
        # save dataframes as tbl's
        df_ICF.to_csv(new_ICF, sep=' ', index=False, header=False)
        df_CEVPF.to_csv(new_CEVPF, sep=' ', index=False, header=False)
        df_FC.to_csv(new_FC, sep=' ', index=False, header=False)
        df_BetaSeepage.to_csv(new_beta, sep=' ', index=False, header=False)
        df_LP.to_csv(new_LP, sep=' ', index=False, header=False)         
        df_PERC.to_csv(new_PERC, sep=' ', index=False, header=False)         
        df_Cflux.to_csv(new_Cflux, sep=' ', index=False, header=False)
        df_SUZ.to_csv(new_SUZ, sep=' ', index=False, header=False)
        df_K4.to_csv(new_K4, sep=' ', index=False, header=False)
        df_K0.to_csv(new_K0, sep=' ', index=False, header=False)
        df_KQuickFlow.to_csv(new_KQuickFlow, sep=' ', index=False, header=False)
        df_N.to_csv(new_N, sep=' ', index=False, header=False)        
        df_N_River.to_csv(new_N_River, sep=' ', index=False, header=False)
        
        
        ### CREATE NEW MODEL
        # Get Model
        #from grpc4bmi.bmi_client_docker import BmiClientDocker
        wflow_Volta_hbv = BmiClientDocker(image='ewatercycle/wflow-grpc4bmi:latest', image_port=55555, 
                                        input_dir=str(output_name),
                                        output_dir="./output")
        
        
        # Initialize model
        wflow_Volta_hbv.initialize('wflow_hbv.ini')
        
        
        ## RUN MODEL
        # Run model for 10+ years and store the LowerZoneStorage, UpperZoneStorage, Soil Moisture and Evapotranspiration
        # GRACE simulations are saved from the start of 2002
        # ET and SM simulations are saved from the start of 2000

        TWS, ET, SM = [], [], []
        for i in range(3711):
            wflow_Volta_hbv.update()

            if wflow_Volta_hbv.get_current_time() >= + wflow_Volta_hbv.get_start_time() + ((365 * 5) + 1) * 86400:
                TWS.append(var_to_xarray_GRACE(wflow_Volta_hbv, variable1="LowerZoneStorage", variable2="UpperZoneStorage", variable3="SoilMoisture"))

#             if wflow_Volta_hbv.get_current_time() >= + wflow_Volta_hbv.get_start_time() + 365 * 3 * 86400:
#                 ET.append(var_to_xarray_ET(wflow_Volta_hbv, variable="ActEvap"))
#                 SM.append(var_to_xarray_SM(wflow_Volta_hbv, variable="SoilMoisture"))

        # Make xarray concatenated datasets from the simulated data, resample the TWS to monthly means and fill months without observations with nans
        TWS = xr.concat(TWS, dim='time')
        TWS = TWS.resample(time='1M').mean()
        TWS[0, :, :] = np.zeros((11, 9)) * np.nan
        TWS[1, :, :] = np.zeros((11, 9)) * np.nan
        TWS[2, :, :] = np.zeros((11, 9)) * np.nan
        TWS[5, :, :] = np.zeros((11, 9)) * np.nan
        TWS[6, :, :] = np.zeros((11, 9)) * np.nan
        TWS[17, :, :] = np.zeros((11, 9)) * np.nan
        TWSA = TWS - TWS.mean(axis=0)
#         ET = xr.concat(ET, dim='time')
#         SM = xr.concat(SM, dim='time')

        # Save xarray datasets as netcdf's
        TWSA.to_netcdf(str(output_folder) + '/TWSA_sim.nc', encoding={'TWSA':{"zlib": True, "complevel": 4}})
#         ET.to_netcdf(str(output_folder) + '/ET_sim.nc', encoding={'ET':{"zlib": True, "complevel": 4}})
#         SM.to_netcdf(str(output_folder) + '/SM_sim.nc', encoding={'SM':{"zlib": True, "complevel": 4}})

        
        ### FINALIZE MODEL AND DELETE NOT NEEDED OUTPUT
        # clean container
        wflow_Volta_hbv.finalize()
        del wflow_Volta_hbv
        
        
        # Assess performance TWSA 
        loc_TWSA_sim = str(output_folder) + '/TWSA_sim.nc'
        Esp_TWSA_mean_cal, Esp_TWSA_mean_val = TWSA_assessment(loc_TWSA_sim)
        
        # Save results TWSA in textfile in column 1
        RS_results = np.loadtxt(str(output_folder) + '/RS_results.txt')
        RS_results[0, 0], RS_results[1, 0] = Esp_TWSA_mean_cal, Esp_TWSA_mean_val
        np.savetxt(str(output_folder) + '/RS_results.txt', RS_results)
        
        # get streamflow csv and put in the RS assessment results in the first columns which is not used!
        sim = pd.read_csv(str(output_folder) + '/run_default/Runoff_gauges.csv', index_col='# Timestep') 
        sim = sim.iloc[:, 0:].to_numpy()
        sim[0, 0], sim[1, 0] = Esp_TWSA_mean_cal, Esp_TWSA_mean_val
        sim = pd.DataFrame(sim)
        sim.to_csv(str(output_folder) + '/run_default/Runoff_gauges.csv')
        sim = sim.iloc[:, 0:].to_numpy()
        print(sim)
                                                       
        # Delete files that are not needed
        shutil.rmtree(str(output_name) + '/inmaps')
        shutil.rmtree(str(output_name) + '/instate')
        shutil.rmtree(str(output_name) + '/intbl')
        shutil.rmtree(str(output_name) + '/intss')
        shutil.rmtree(str(output_name) + '/staticmaps')
        
        # return streamflow
        return sim

        
    def evaluation(self):
        ### GET EVALUATION DATA OF THE MODEL
        eva = pd.read_csv('wflow_Volta_hbv2_0/intss/Streamflow_calibration_data.csv', delimiter=';', skiprows=0, index_col='Date')
        eva[eva==-9999.00] = np.nan
        eva = eva.iloc[:, 1:].to_numpy()
        return eva

    def objectivefunction(self, evaluation, simulation, params=None):
        ###
        # Get monthly rainfall data for runoff calculation coefficients
        P_Rc = pd.read_csv('wflow_Volta_hbv2_0/intss/monthly_P_arrays_per_subcatchment.csv', delimiter=',', index_col='Unnamed: 0')
        P_Rc = P_Rc.to_numpy()
        
        #SPOTPY expects to get one or multiple values back, that define the performence of the model run
        if not self.obj_func:
            # Return results of calibration and validation period (shape is mean_KGE(5), KGE, KGE_BC, KGE_FDC, KGE_ACF, KGE_Rc)
            RS_results = simulation[0:6, 0]
            simulation = simulation[:, 1:]
            
            # Results calibration catchment BV
            res_cal1, res_val1 = streamflow_performance_func(evaluation[:, 0], simulation[:, 0], P_Rc[0, :])
            res_cal2, res_val2 = streamflow_performance_func(evaluation[:, 1], simulation[:, 1], P_Rc[1, :])
            res_cal3, res_val3 = streamflow_performance_func(evaluation[:, 2], simulation[:, 2], P_Rc[2, :])
            res_cal4, res_val4 = streamflow_performance_func(evaluation[:, 3], simulation[:, 3], P_Rc[3, :])
            res_cal5, res_val5 = streamflow_performance_func(evaluation[:, 4], simulation[:, 4], P_Rc[4, :])
            
            # Results validaton catchment WV
            res_cal6, res_val6 = streamflow_performance_func(evaluation[:, 5], simulation[:, 5], P_Rc[5, :])
            res_cal7, res_val7 = streamflow_performance_func(evaluation[:, 6], simulation[:, 6], P_Rc[6, :])
            res_cal8, res_val8 = streamflow_performance_func(evaluation[:, 7], simulation[:, 7], P_Rc[7, :])
            res_cal9, res_val9 = streamflow_performance_func(evaluation[:, 8], simulation[:, 8], P_Rc[8, :])
            
            # Results calibration catchment Oti
            res_cal10, res_val10 = streamflow_performance_func(evaluation[:, 9], simulation[:, 9], P_Rc[9, :])
            res_cal11, res_val11 = streamflow_performance_func(evaluation[:, 10], simulation[:, 10], P_Rc[10, :])
            res_cal12, res_val12 = streamflow_performance_func(evaluation[:, 11], simulation[:, 11], P_Rc[11, :])  
            
            
            # Calibration catchment calibration period
            mean_mean_cal    = np.mean(np.array([res_cal1[0], res_cal2[0], res_cal3[0], res_cal4[0], res_cal5[0], res_cal10[0], res_cal11[0], res_cal12[0]]))
            mean_KGE_cal     = np.mean(np.array([res_cal1[1], res_cal2[1], res_cal3[1], res_cal4[1], res_cal5[1], res_cal10[1], res_cal11[1], res_cal12[1]]))
            mean_KGE_BC_cal  = np.mean(np.array([res_cal1[2], res_cal2[2], res_cal3[2], res_cal4[2], res_cal5[2], res_cal10[2], res_cal11[2], res_cal12[2]]))
            mean_KGE_FDC_cal = np.mean(np.array([res_cal1[3], res_cal2[3], res_cal3[3], res_cal4[3], res_cal5[3], res_cal10[3], res_cal11[3], res_cal12[3]]))
            mean_KGE_ACF_cal = np.mean(np.array([res_cal1[4], res_cal2[4], res_cal3[4], res_cal4[4], res_cal5[4], res_cal10[4], res_cal11[4], res_cal12[4]]))
            mean_KGE_Rc_cal  = np.mean(np.array([res_cal1[5], res_cal2[5], res_cal3[5], res_cal4[5], res_cal5[5], res_cal10[5], res_cal11[5], res_cal12[5]]))
            
            # Calibration catchment validation period
            mean_mean_val    = np.mean(np.array([res_val1[0], res_val2[0], res_val3[0], res_val4[0], res_val5[0], res_val10[0], res_val11[0], res_val12[0]]))
            mean_KGE_val     = np.mean(np.array([res_val1[1], res_val2[1], res_val3[1], res_val4[1], res_val5[1], res_val10[1], res_val11[1], res_val12[1]]))
            mean_KGE_BC_val  = np.mean(np.array([res_val1[2], res_val2[2], res_val3[2], res_val4[2], res_val5[2], res_val10[2], res_val11[2], res_val12[2]]))
            mean_KGE_FDC_val = np.mean(np.array([res_val1[3], res_val2[3], res_val3[3], res_val4[3], res_val5[3], res_val10[3], res_val11[3], res_val12[3]]))
            mean_KGE_ACF_val = np.mean(np.array([res_val1[4], res_val2[4], res_val3[4], res_val4[4], res_val5[4], res_val10[4], res_val11[4], res_val12[4]]))
            mean_KGE_Rc_val  = np.mean(np.array([res_val1[5], res_val2[5], res_val3[5], res_val4[5], res_val5[5],  res_val10[5], res_val11[5], res_val12[5]]))
            
            # Validation catchment, calibration period
            mean_mean_cal2    = np.mean(np.array([res_cal6[0], res_cal7[0], res_cal8[0], res_cal9[0]]))
            mean_KGE_cal2     = np.mean(np.array([res_cal6[1], res_cal7[1], res_cal8[1], res_cal9[1]]))
            mean_KGE_BC_cal2  = np.mean(np.array([res_cal6[2], res_cal7[2], res_cal8[2], res_cal9[2]]))
            mean_KGE_FDC_cal2 = np.mean(np.array([res_cal6[3], res_cal7[3], res_cal8[3], res_cal9[3]]))
            mean_KGE_ACF_cal2 = np.mean(np.array([res_cal6[4], res_cal7[4], res_cal8[4], res_cal9[4]]))
            mean_KGE_Rc_cal2  = np.mean(np.array([res_cal6[5], res_cal7[5], res_cal8[5], res_cal9[5]]))
            
            # Validation catchment, validation period
            mean_mean_val2    = np.mean(np.array([res_val6[0], res_val7[0], res_val8[0], res_val9[0]]))
            mean_KGE_val2     = np.mean(np.array([res_val6[1], res_val7[1], res_val8[1], res_val9[1]]))
            mean_KGE_BC_val2  = np.mean(np.array([res_val6[2], res_val7[2], res_val8[2], res_val9[2]]))
            mean_KGE_FDC_val2 = np.mean(np.array([res_val6[3], res_val7[3], res_val8[3], res_val9[3]]))
            mean_KGE_ACF_val2 = np.mean(np.array([res_val6[4], res_val7[4], res_val8[4], res_val9[4]]))
            mean_KGE_Rc_val2  = np.mean(np.array([res_val6[5], res_val7[5], res_val8[5], res_val9[5]]))
            
            # Take mean of the performance functions for all streamflow stations
            u = np.array([res_cal1[0], res_cal2[0], res_cal3[0], res_cal4[0], res_cal5[0], res_cal10[0], res_cal11[0], res_cal12[0]]) # station 6, 7, 8 and 9 are used for validation
            mean_KGE_cal_stations = np.mean(u)
            
            # save results in array with shape (14, 13) (mean_KGE_cal_stations, cal results(6), val_results(6) for calibration and validation catchments)
            # create mean_cal and mean_val arrays
            means_cal = np.array([mean_mean_cal, mean_KGE_cal, mean_KGE_BC_cal, mean_KGE_FDC_cal, mean_KGE_ACF_cal, mean_KGE_Rc_cal])
            means_val = np.array([mean_mean_val, mean_KGE_val, mean_KGE_BC_val, mean_KGE_FDC_val, mean_KGE_ACF_val, mean_KGE_Rc_val])
            means_cal2 = np.array([mean_mean_cal2, mean_KGE_cal2, mean_KGE_BC_cal2, mean_KGE_FDC_cal2, mean_KGE_ACF_cal2, mean_KGE_Rc_cal2])
            means_val2 = np.array([mean_mean_val2, mean_KGE_val2, mean_KGE_BC_val2, mean_KGE_FDC_val2, mean_KGE_ACF_val2, mean_KGE_Rc_val2])
            
            # create the array to save
            data = np.zeros((1, 14, 13))
            data[0, :, 0] = mean_KGE_cal_stations
            data[0, 0, 1:7], data[0, 0, 7:] = res_cal1, res_val1
            data[0, 1, 1:7], data[0, 1, 7:] = res_cal2, res_val2
            data[0, 2, 1:7], data[0, 2, 7:] = res_cal3, res_val3
            data[0, 3, 1:7], data[0, 3, 7:] = res_cal4, res_val4
            data[0, 4, 1:7], data[0, 4, 7:] = res_cal5, res_val5
            data[0, 5, 1:7], data[0, 5, 7:] = res_cal6, res_val6
            data[0, 6, 1:7], data[0, 6, 7:] = res_cal7, res_val7
            data[0, 7, 1:7], data[0, 7, 7:] = res_cal8, res_val8
            data[0, 8, 1:7], data[0, 8, 7:] = res_cal9, res_val9
            data[0, 9, 1:7], data[0, 9, 7:] = res_cal10, res_val10
            data[0, 10, 1:7], data[0, 10, 7:] = res_cal11, res_val11
            data[0, 11, 1:7], data[0, 11, 7:] = res_cal12, res_val12
            data[0, 12, 1:7], data[0, 12, 7:] = means_cal, means_val
            data[0, 13, 1:7], data[0, 13, 7:] = means_cal2, means_val2
            
            # Save results, both in total and separate file per run
            results = np.array((np.loadtxt('Results_DDS.txt')))
            results = results.reshape(int(np.shape(results)[0]), 14, 13)
            results = np.append(results, data, axis=0)
            results = results.reshape(results.shape[0], 182)
            np.savetxt('Results_DDS.txt', results)
            
            # load TWSA results and caculate mean of mean Esp TWSA and mean of stations of KGE (which is a mean of 5 KGE's)
            mean_Esp_TWSA = RS_results[0]
            mean_Esp_ET   = RS_results[2]
            mean_Esp_SM   = RS_results[4]
            
            like = np.mean([mean_KGE_cal_stations, mean_Esp_TWSA])
            
            
        else:
            # THIS PART IS NOT USED
            #Way to ensure on flexible spot setup class
            ED_1 = streamflow_performance_func(evaluation[:, 0], simulation[:, 0], P_Rc[0, :])
            ED_2 = streamflow_performance_func(evaluation[:, 1], simulation[:, 1], P_Rc[1, :])
            ED_3 = streamflow_performance_func(evaluation[:, 2], simulation[:, 2], P_Rc[2, :])
            ED_4 = streamflow_performance_func(evaluation[:, 3], simulation[:, 3], P_Rc[3, :])
            ED_5 = streamflow_performance_func(evaluation[:, 4], simulation[:, 4], P_Rc[4, :])
            ED_6 = streamflow_performance_func(evaluation[:, 5], simulation[:, 5], P_Rc[5, :])
            ED_7 = streamflow_performance_func(evaluation[:, 6], simulation[:, 6], P_Rc[6, :])
            ED_8 = streamflow_performance_func(evaluation[:, 7], simulation[:, 7], P_Rc[7, :])
            ED_9 = streamflow_performance_func(evaluation[:, 8], simulation[:, 8], P_Rc[8, :])
            ED_10 = streamflow_performance_func(evaluation[:, 9], simulation[:, 9], P_Rc[9, :])
            ED_11 = streamflow_performance_func(evaluation[:, 10], simulation[:, 10], P_Rc[10, :])
            ED_12 = streamflow_performance_func(evaluation[:, 11], simulation[:, 11], P_Rc[11, :])       
            
            # Take ED of the performance functions for all streamflow stations
            u = np.array([ED_1[0], ED_2[0], ED_3[0], ED_4[0], ED_5[0], ED_6[0], ED_7[0], ED_8[0], ED_9[0], ED_10[0], ED_11[0], ED_12[0]])
            v = np.ones(12)
            like = -distance.euclidean(u, v)
            
        return like
