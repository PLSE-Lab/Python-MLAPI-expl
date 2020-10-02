import pandas as pd
import numpy as np
import random
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 

FILE="../input/accident.csv"
d=pd.read_csv(FILE)

FILE="../input/pbtype.csv"
b=pd.read_csv(FILE)

FILE="../input/person.csv"
person=pd.read_csv(FILE)

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


db = pd.merge(d, b, how='right',left_on='ST_CASE', right_on='ST_CASE')

per = person[person['PER_TYP']==6][['ST_CASE','PER_NO','STR_VEH','DEATH_TM']]

# Throw this back in d
d = pd.merge(per, db, how='left',left_on=['ST_CASE','PER_NO'], right_on=['ST_CASE','PER_NO'])

# Throw this back in d
#d = db[db['PBPTYPE']==6][['ST_CASE','crashTime','PER_NO','LATITUDE','LONGITUD','FATALS','DRUNK_DR']]

# db['PBPTYPE']==6


# Set index
# d.index = pd.DatetimeIndex(d.timeStamp)



# Creating an HTML HEADER FILE
headV="""<!DOCTYPE html>
<html>
  <head>
  <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>montcoalert.org</title>
    <style>
      html, body {
      height: 100%;
      margin: 0;
      padding: 0;
      }
      #map {
      height: 100%;
      }
    </style>
  </head>
  <body> 
      

      <!--  DataCanary_s fix -->
      <div id="map" class="main-container"></div>
    <script>

      function initMap() {
      var map = new google.maps.Map(document.getElementById('map'), {
        zoom: 5,
        center: {lat: 40.069244, lng: -93.3626537}
      
      });
      	 
      setMarkers(map);
      // Add traffic
      trafficLayer = new google.maps.TrafficLayer();
	  trafficLayer.setMap(map);	
	  
	  // Add bikeLayer
	  var bikeLayer = new google.maps.BicyclingLayer();
      bikeLayer.setMap(map);
	  
      }
"""

tailV="""      function setMarkers(map) {
      // Adds markers to the map.

      // Marker sizes are expressed as a Size of X,Y where the origin of the image
      // (0,0) is located in the top left of the image.

      // Origins, anchor positions and coordinates of the marker increase in the X
      // direction to the right and in the Y direction down.
      var image = {
            url: 'https://storage.googleapis.com/montco-stats/images/carCrash.png',

      // This marker is 20 pixels wide by 32 pixels high.
      size: new google.maps.Size(20, 32),
      // The origin for this image is (0, 0).
      origin: new google.maps.Point(0, 0),
      // The anchor for this image is the base of the flagpole at (0, 32).
      anchor: new google.maps.Point(0, 32)
      };
      // Shapes define the clickable region of the icon. The type defines an HTML
      // <area> element 'poly' which traces out a polygon as a series of X,Y points.
// The final coordinate closes the poly by connecting to the first coordinate.

      function htmlEntities(str) {
//         return String(str).replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
         return String(str).replace(/>/g, '&gt;').replace(/"/g, '&quot;');
       }

      var shape = {
      coords: [1, 1, 1, 20, 18, 20, 18, 1],
      type: 'poly'
      };
      
       for (var i = 0; i < crashes.length; i++) {
                          var crash = crashes[i];
                          var marker = new google.maps.Marker({
                          position: {lat: crash[1], lng: crash[2]},
                          map: map,
                          icon: crash[4],
                          shape: shape,
                          draggable: true,
                          title: htmlEntities(crash[0]),
                          zIndex: crash[3]
                          });
                          }
                          }

                          </script>

        <script async defer
            src="https://maps.googleapis.com/maps/api/js?key=AIzaSyA0wJsknjKk5pkO2aOqsIGkSNcELPjc830&signed_in=true&callback=initMap"></script>

  </body>
</html>
      
""" 
s=' var crashes = [\n'


#  **  SELECT SECTION **





# So many points at exact location
def myRand():
    r=random.random()*0.001-0.0005
    return r

#d.lat=d.lat.apply(lambda x: str(float(x)+myRand()))
#d.lng=d.lng.apply(lambda x: str(float(x)+myRand()))





t=d[d['DEATH_TM']!=8888]
title=[]
for i in t.ST_CASE.tolist():
    title.append("ST_CASE:%d" % i)

desc=[]
for i in t.PER_NO.tolist():
    desc.append("PER_NO:%d" % i)

twp=[]
for i in t.DEATH_TM.tolist():
    twp.append("DEATH_TM:%d" % i)


timeStamp=t.crashTime.tolist()
lat=t.LATITUDE.tolist()
lng=t.LONGITUD.tolist()
for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s ', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/bikeKilled.png'],\n" % (displayTitle,lat[i],lng[i],i)


t=d[d['DEATH_TM']==8888]
title=t.ST_CASE.tolist()
desc=t.PER_NO.tolist()
twp=t.DEATH_TM.tolist()
timeStamp=t.crashTime.tolist()
lat=t.LATITUDE.tolist()
lng=t.LONGITUD.tolist()
for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s ', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/bike.png'],\n" % (displayTitle,lat[i],lng[i],i)




s+='];'

# Write out 
f=open('__results__.html','w')
f.write(headV)
f.write(s)
f.write(tailV)
f.close()

# Write out 
f=open('output.html','w')
f.write(headV)
f.write(s)
f.write(tailV)
f.close()

