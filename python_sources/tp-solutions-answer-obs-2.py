def obs_answer_2(year,date,station_id,param):
    year = date[0:4]
    fname = '/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+year+".csv"
    #df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
    study_date = pd.Timestamp(date)  #study date
    d_sub = df[(df['date'] == study_date) & (df['lat'] == lat) & (df['lon'] == lon)]
    display(d_sub)
    d_sub_param = d_sub[param]
    display(d_sub_param)
    return d_sub_param