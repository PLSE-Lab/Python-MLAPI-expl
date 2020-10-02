def open_and_select(run_date,param,model,lat,lon,step):
    #open the corresponding file according to the chosen parameter
    if param == 't2m' or param == 'd2m' or param == 'r':
        level = '2m'
    elif param == 'ws' or param =='p3031' or param == 'u10' or param == 'v10':
        level = '10m'
    elif param == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'

    year = run_date[0:4]
    month = run_date[5:7]
    day = run_date[8:10]

    directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + year + month + '/' + year + month + '/'
    fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{year}{month}{day}000000.nc'
    dm = xr.open_dataset(fname)

    #select the searched value
    result = dm.sel(latitude=lat, longitude=lon, method='nearest').isel(step = step)[param]
    print('run date',result['time'].values)
    print('nearest point latitude',result['latitude'].values)
    print('nearest point longitude',result['longitude'].values)
    print('step date',result['valid_time'].values)
    print('parameter name',result.name)
    print('parameter value',result.values)
    print('data overview',result)
    return result