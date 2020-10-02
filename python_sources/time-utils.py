import datetime
import pandas as pd
import numbers

HOUR_SECONDS = 3600.0

def time_interval_months(time_serie, str_format="%Y-%m-%d"):
    delta_h = delta_time_hour(time_serie.tolist()[0], 
                                         time_serie.tolist()[-1], 
                                         str_format)
    return delta_h / ((24*365)/12)

def str_to_datetime(str_dt, str_dt_format='%d/%m/%Y %H:%M:%S'):
    '''
    Returns a datetime.datetime object
    :param str_dt: string representing date and/or time
    :param str_dt_format:
    :return:
    '''
    use_str_dt = str_dt.split(".")[0]
    return datetime.datetime.strptime(use_str_dt, str_dt_format)

def time_resolution(data):
    dates = data["Date"].drop_duplicates().apply(pd.to_datetime).sort_values()
    return dates.iloc[1: len(dates) - 1].reset_index(drop=True) - dates.iloc[0: len(dates) - 2].reset_index(
        drop=True)

def delta_time_seconds(str_dt_initial, str_dt_final, str_dt_format='%Y-%m-%d %H:%M:%S'):
    '''
    Returns the time difference (seconds) between two strings representing date and/time
    :param str_dt_initial: string representing date and/or time
    :param str_dt_final: string representing date and/or time
    :return:
    '''

    if type(str_dt_initial) is float or type(str_dt_initial) is int or isinstance(str_dt_initial, numbers.Number):
        str_dt_initial = int(round(str_dt_initial))
        if len(str(str_dt_initial)) == 13:
            str_dt_initial = timestamp_to_datetime(str_dt_initial / 1000)
        elif len(str(str_dt_initial)) == 10:
            str_dt_initial = timestamp_to_datetime(str_dt_initial)

    elif (not type(str_dt_initial) is datetime.datetime) and (not type(str_dt_initial) is pd.Timestamp):
        str_dt_initial = str_to_datetime(str_dt_initial, str_dt_format)


    if type(str_dt_final) is float or type(str_dt_final) is int or isinstance(str_dt_final, numbers.Number):
        str_dt_final = int(round(str_dt_final))
        if len(str(str_dt_final)) == 13:
            str_dt_final = timestamp_to_datetime(str_dt_final / 1000)
        elif len(str(str_dt_final)) == 10:
            str_dt_final = timestamp_to_datetime(str_dt_final)

    elif (not type(str_dt_final) is datetime.datetime) and (not type(str_dt_final) is pd.Timestamp):
        str_dt_final = str_to_datetime(str_dt_final, str_dt_format)

    return abs((str_dt_final - str_dt_initial).total_seconds())

def str_datetime_to_timestamp(str_datetime, str_dt_format='%Y-%m-%d %H:%M:%S'):
    use_str_dt = str_datetime.split(".")[0]
    dt = str_to_datetime(use_str_dt, str_dt_format)
    return datetime.datetime.timestamp(dt)

def timestamp_to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts)

def delta_time_hour(str_dt_initial, str_dt_final, str_dt_format='%Y-%m-%d %H:%M:%S'):
    return delta_time_seconds(str_dt_initial, str_dt_final, str_dt_format=str_dt_format) / HOUR_SECONDS
