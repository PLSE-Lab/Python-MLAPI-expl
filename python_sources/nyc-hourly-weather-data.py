import requests
import sys
import pandas
import time
from datetime import date, timedelta
import socket

s = socket.socket()
# s.connect(('google.com', 80))
s.connect(('api.wunderground.com/api/5b986e87deb169bd/history_20160101/q/CA/San_Francisco.json', 80))
def delay(secs):
    for i in range(secs,0,-1):
        time.sleep(1)
        sys.stdout.write('#'+' ')
        sys.stdout.flush()

# Required to query Wunderground.com's API
Key = "5b986e87deb169bd" #
State = "NY" # Default State
City = "New_York" # Default City
DateFrom = date(2016, 12, 31)
DateTo = date(2017, 1, 1)
csvPath = '/Users/Meinertsen/PycharmProjects/Weather/'
delta = DateTo - DateFrom # timedelta
gethead = {
    'User-Agent': 'Mozilla/5.0'
}

ldate, qhead, rhead, shead = [], [], [], []
for i in range(delta.days + 1):
    ldate.append(DateFrom + timedelta(days=i))
# print(pandas.DataFrame(ldate))
# delay(7) # Delay for 7 secs to comply with Wundergrounds API rulesx
for idate in range(0,len(ldate)):
    YYYY = ldate[idate].year
    MM = ldate[idate].month
    DD = ldate[idate].day
    delay(7)
    print('Processing '+str(YYYY) + '-' + str(MM).zfill(2) + '-' + str(DD).zfill(2))
    urlapi = "http://api.wunderground.com/api/" + Key + "/history_" + \
             str(YYYY) + str(MM).zfill(2) + str(DD).zfill(2) + "/q/" + \
             State + "/" + City + ".json"
    # print(urlapi)  # Print if you want to see the complete URL to e.g. use with http://jsoneditoronline.org
    json_data=requests.get(urlapi, headers=gethead).json() # Use requests to get JSON
    urlapi=""
    for i in enumerate(json_data.keys()):
        for j in enumerate(json_data[i[1]].keys()):
            # print("\t" + j[1])
            if i[0] == 1 and j[0] == 2:  # Checks if equal to 'observations'
                for ihead in enumerate(json_data[i[1]][j[1]][0]):  # Grab headers for hourly observations
                    if ihead[0]<2:
                        lhead = ['Datetime'] # List of headers starts with Datetime
                    elif ihead[0]>(len(json_data[i[1]][j[1]][0])-2):
                        dhead = []
                        dhead.append(lhead)
                        while idate==0:
                            pandas.DataFrame(qhead, columns=lhead).to_csv(csvPath + str("Weather") + '.csv', index=False, encoding='utf-8')
                            break
                    else:
                        lhead.append(ihead[1])
                for k in range(0, len(json_data[i[1]][j[1]])):  # 24 List hourly observations
                    l = json_data[i[1]][j[1]][k]
                    m = l['date']
                    phead = [m['year'] + '-' +m['mon'] + '-' +m['mday'] + ' ' +m['hour'] + ':' + m['min']]
                    for hcount in range(1, len(lhead)):
                        phead.append(l[lhead[hcount]])
                    qhead.append(phead)
                qhead = pandas.DataFrame(qhead, columns=lhead)
                rhead = pandas.DataFrame(pandas.read_csv(csvPath + "Weather" + ".csv"), columns=lhead)
                rhead = pandas.concat([rhead, qhead],ignore_index=True, axis=0)
                pandas.DataFrame(rhead, columns=lhead).to_csv(csvPath + "WeatherData" + ".csv", header=True, index=False, encoding='utf-8')
                qhead = []
            else: pass

# Loop through JSON data to see Key structure (For viewing purpose only)
for i in enumerate(json_data.keys()):   # Level (i) JSON Keys
    try:
        print("(" + str(i[0]) + ") " + i[1])
        for j in enumerate(json_data[i[1]].keys()): # Level (j) JSON Keys
            try:
                print("\t" + "(" + str(j[0]) + ") " + j[1])
                for k in enumerate(json_data[i[1]][j[1]].keys()): # Level (k) JSON Keys
                    try:
                        print("\t" * 2 + "(" + str(k[0]) + ") " + k[1])
                    except: pass
            except: pass
    except: pass