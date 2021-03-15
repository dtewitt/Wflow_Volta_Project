def var_to_xarray(model, variable):
    from cftime import num2date
    import numpy as np
    import xarray as xr
    # Get grid properties from model (x = latitude !!)
    # could be speedup, lots of bmi calls are done here that dont change between updates
    shape = model.get_grid_shape(model.get_var_grid(variable))
    lat = model.get_grid_x(model.get_var_grid(variable))
    lon = model.get_grid_y(model.get_var_grid(variable))
    time = num2date(model.get_current_time(), model.get_time_units())

    # Get model data for variable at current timestep
    data = model.get_value(variable)
    data = np.reshape(data, shape)

    # Create xarray object
    da = xr.DataArray(data, Rc_obs_val, 
                      coords = {'longitude': lon, 'latitude': lat, 'time': time}, 
                      dims = ['latitude', 'longitude'],
                      name = variable,
                      attrs = {'units': model.get_var_units(variable)}
                     )

    # Masked invalid values on return array:
    return da.where(da != -999)


def var_to_xarray_GRACE(model, variable1, variable2, variable3):
    # Get grid properties from model (x = latitude !!)
    # could be speedup, lots of bmi calls are done here that dont change between updates
    import numpy as np
    import xarray as xr
    from cftime import num2date
    
    # Get model shape, lat, lon and time coordinates
    shape = (220, 180)
    lat = np.linspace(5.5, 15.5, 11)
    lon = np.linspace(-5.5, 2.5, 9)
    time = num2date(model.get_current_time(), model.get_time_units())    
    
    # Get model data for variable 1 at current timestep
    data_LZS = model.get_value(variable1)
    data_LZS = np.reshape(data_LZS, shape)
    data_LZS[data_LZS==-999]=np.nan
    
    # Get model data for variable 2 at current timestep
    data_UZS = model.get_value(variable2)
    data_UZS = np.reshape(data_UZS, shape)
    data_UZS[data_UZS==-999]=np.nan  
    
    # Get model data for variable 3 at current timestep
    data_SM = model.get_value(variable3)
    data_SM = np.reshape(data_SM, shape)
    data_SM[data_SM==-999]=np.nan
    
    # Check for cells with more or less than half nan values, if more, than make all values nan-values
    for i in range(11):
        y = i * 20
        for j in range(9):
            x = j * 20
            
            # Count number of nan's, if it is higher than half, alleverything is now nan
            nancount_LZS = np.count_nonzero(np.isnan(data_LZS[y:y+20, x:x+20]))
            nancount_UZS = np.count_nonzero(np.isnan(data_UZS[y:y+20, x:x+20]))
            nancount_SM  = np.count_nonzero(np.isnan(data_LZS[y:y+20, x:x+20]))
            
            part_nan_LZS = nancount_LZS / (20 * 20)
            part_nan_UZS = nancount_UZS / (20 * 20)
            part_nan_SM  = nancount_SM  / (20 * 20)
            
            if part_nan_LZS > 0.5:
                data_LZS[y:y+20, x:x+20] = np.nan
            if part_nan_UZS > 0.5:  
                data_UZS[y:y+20, x:x+20] = np.nan
            if part_nan_SM > 0.5:
                data_SM[y:y+20, x:x+20] = np.nan
                
    # reshape and take mean horizontally variable 1, 2 and 3
    data_LZS = np.reshape(data_LZS, (220, 9, 20))
    data_LZS = np.nanmean(data_LZS, axis=2)
    
    data_UZS = np.reshape(data_UZS, (220, 9, 20))
    data_UZS = np.nanmean(data_UZS, axis=2)
    
    data_SM = np.reshape(data_SM, (220, 9, 20))
    data_SM = np.nanmean(data_SM, axis=2)

    # reshape and take mean vertically variable 1, 2 and 3
    data_LZS = np.reshape(data_LZS, (11, 20, 9))
    data_LZS = np.nanmean(data_LZS, axis=1)

    data_UZS = np.reshape(data_UZS, (11, 20, 9))
    data_UZS = np.nanmean(data_UZS, axis=1)

    data_SM = np.reshape(data_SM, (11, 20, 9))
    data_SM = np.nanmean(data_SM, axis=1)

    # Sum all the xarray datasets to get simulated TWS
    TWS = data_LZS + data_UZS + data_SM
    
    # Create xarray object
    da = xr.DataArray(TWS, 
                      coords = {'longitude': lon, 'latitude': lat, 'time': time}, 
                      dims = ['latitude', 'longitude'],
                      name = 'TWSA',
                      attrs = {'units': 'mm'})

    # Masked invalid values on return array:
    return da.where(da != -999)


def var_to_xarray_ET(model, variable):
    # Get grid properties from model (x = latitude !!)
    # could be speedup, lots of bmi calls are done here that dont change between updates
    import numpy as np
    import xarray as xr
    from cftime import num2date
    
    # Get model shape, lat, lon and time coordinates
    shape = (220, 180)
    lat = np.linspace(5.025, 15.975, 220)
    lon = np.linspace(-5.975, 2.975, 180)
    time = num2date(model.get_current_time(), model.get_time_units())    
    
    # Get model data for variable at current timestep
    data_ET = model.get_value(variable)
    data_ET = np.reshape(data_ET, shape)
    data_ET[data_ET==-999]=np.nan
    
    # Create xarray object
    da = xr.DataArray(data_ET, 
                      coords = {'longitude': lon, 'latitude': lat, 'time': time}, 
                      dims = ['latitude', 'longitude'],
                      name = 'ET',
                      attrs = {'units': 'mm'}
                     )

    # Masked invalid values on return array:
    return da.where(da != -999)


def var_to_xarray_SM(model, variable):
    # Get grid properties from model (x = latitude !!)
    # could be speedup, lots of bmi calls are done here that dont change between updates
    import numpy as np
    import xarray as xr
    from cftime import num2date
    
    # Get model shape, lat, lon and time coordinates
    shape = (220, 180)
    lat = np.linspace(5.125, 15.875, 44)
    lon = np.linspace(-5.875, 2.875, 36)
    time = num2date(model.get_current_time(), model.get_time_units())    
    
    # Get model data for variable at current timestep
    data_SM = model.get_value(variable)
    data_SM = np.reshape(data_SM, shape)
    data_SM[data_SM==-999]=np.nan

    
    # reshape and take mean horizontally of variable 
    data_SM = np.reshape(data_SM, (220, 36, 5))
    data_SM = np.mean(data_SM, axis=2)
    
    # reshape and take mean vertically of variable 
    data_SM = np.reshape(data_SM, (44, 5, 36))
    data_SM = np.mean(data_SM, axis=1)
    
    
    # Create xarray object
    da = xr.DataArray(data_SM, 
                      coords = {'longitude': lon, 'latitude': lat, 'time': time}, 
                      dims = ['latitude', 'longitude'],
                      name = 'SM',
                      attrs = {'units': 'mm'})

    # Masked invalid values on return array:
    return da.where(da != -999)