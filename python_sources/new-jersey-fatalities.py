import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read/clean data 

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

FILE="../input/accident.csv"

d=pd.read_csv(FILE)

def f(x):
    year = x[0]
    month = x[1]
    day = x[2]
    hour = x[3]
    minute = x[4]
    # Sometimes they don't know hour and minute
    if hour == 99:
        hour = 0
    if minute == 99:
        minute = 0
    s = "%02d-%02d-%02d %02d:%02d:00" % (year,month,day,hour,minute)
    c = datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
    return c
 
d['crashTime']   = d[['YEAR','MONTH','DAY','HOUR','MINUTE']].apply(f, axis=1)
d['crashDay']    = d['crashTime'].apply(lambda x: x.date())
d['crashMonth']  = d['crashTime'].apply(lambda x: x.strftime("%B") )
d['crashMonthN'] = d['crashTime'].apply(lambda x: x.strftime("%d") ) # sorting
d['crashTime'].head()


headV="""
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>Marker Clustering</title>
    <style>
      /* Always set the map height explicitly to define the size of the div
       * element that contains the map. */
      #map {
        height: 100%;
      }
      /* Optional: Makes the sample page fill the window. */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    <!--  DataCanary_s fix -->
    <div id="map" class="main-container"></div>
    
    <script>

      function initMap() {

        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 4,
          center: {lat: 38.4772596, lng: -105.2578757}
        });

        // Create an array of alphabetical characters used to label the markers.
        var labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        var shape = {
          coords: [1, 1, 1, 20, 18, 20, 18, 1],
          type: 'poly'
        };

        // Add some markers to the map.
        // Note: The code uses the JavaScript Array.prototype.map() method to
        // create an array of markers based on a given "locations" array.
        // The map() method here has nothing to do with the Google Maps API.
        var markers = crashes.map(function(crash, i) {
          return new google.maps.Marker({
            position: {lat: crash[1], lng: crash[2]},
            label: labels[i % labels.length],
            icon: crash[4],
            shape: shape,
            draggable: true,
            title: crash[0],

          });
        });

        // Add a marker clusterer to manage the markers.
        var markerCluster = new MarkerClusterer(map, markers,
            {imagePath: 'https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/m'});
      }
      
      """
s=' var crashes = [\n'




d=d[(d.STATE == 34)]
# Missing some lat/lon
# If nan's get on the graph, it won't display.
d.fillna(0, inplace=True)
d=d[(d.LONGITUD != 0 ) | (d.LATITUDE != 0 )]


t=d
#t=k[['Text_General_Code','Dispatch_Date_Time','Lon','Lat','Location_Block','Hour']]

# Change these mike...to be meaningful, after working
title=[]
for i in t.ST_CASE.tolist():
    title.append("ST_CASE:%s " % i)
    

desc=[]
for i in t.PERSONS.tolist():
    desc.append("PERSONS:%s " % i)
    

twp=[]
for i in t.FATALS.tolist():
    twp.append("   FATALS:%s" % i)

timeStamp=t.crashTime.tolist()
lat=t.LATITUDE.tolist()
lng=t.LONGITUD.tolist()



for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/homicide.png'],\n" % (displayTitle,lat[i],lng[i],i)

s+='];'


tailV = """
     
    </script>
    <script src="https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/markerclusterer.js">
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=AIzaSyA0wJsknjKk5pkO2aOqsIGkSNcELPjc830&callback=initMap">
    </script>
  </body>
</html>
"""


# Write out 
f=open('__results__.html','w')
f.write(headV)
f.write(s)
f.write(tailV)
f.close()




# Write out (I always do this...easier to debug)
f=open('output.html','w')
f.write(headV)
f.write(s)
f.write(tailV)
f.close()

