# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#os.system('pip install cerberus')
import json
import cerberus as crs
from sys import exit

#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print ('Starting')
sWeatherFileName = '../input/weathermoscowforecast/weather_moscow_20190131.json'
sHolidayFileName  = '../input/ruholiday/day'

with open (sHolidayFileName, 'r') as f:
    h = json.load(f)

with open (sWeatherFileName, 'r') as f:
    w = json.load(f)
    
schemeTop = {
                'consolidated_weather': {'type': 'list'}
                ,'title' : {'type': 'string'}
            }

schemeForecast = {
                     'id': {'type': 'integer'}
                    ,'min_temp': {'type': 'float'}
                    ,'max_temp': {'type': 'float'}
                    ,'the_temp': {'type': 'float'}
                }

schemeHoliday = {
                    'holiday' : {'type': 'boolean'}
                    ,'date' : {'type': 'string', 'regex' : '^\d{4}-\d?\d-\d?\d$'}
                }
                
v = crs.Validator()
validated = []
validated.append(v.validate (h, schemeHoliday))
v.errors

print (validated)
v.allow_unknown = True


validated.append(v.validate (w, schemeTop))
if not all(validated) : exit (1)
for i in range(len(w['consolidated_weather'])) :
    validated.append(v.validate (w['consolidated_weather'][i], schemeForecast))
if not all(validated) : exit (1)
print (validated)