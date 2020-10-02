import pandas as pd
import numpy as np
import random
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:00 %p')

# Read data 
d=pd.read_csv("../input/crime.csv",parse_dates=['incident_datetime'],date_parser=dateparse)



# Set index

d['timeStamp'] = d.incident_datetime
d['desc'] = d.incident_description
#d['desc'] = d['desc'].apply(lambda x:   x.replace('$',''))
d['lat'] = d.latitude
d['lng'] = d.longitude
d['twp'] = d.address_1
d['title'] = d.case_number

d = d[np.isfinite(d['lat'])]
d = d[np.isfinite(d['lng'])]


# k[['title','timeStamp','lat','lng','desc','twp']]


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
        zoom: 13,
        center: {lat: 40.069244, lng: -75.130606}
      
      });
      	 
      setMarkers(map);
      // Add traffic
      trafficLayer = new google.maps.TrafficLayer();
	  trafficLayer.setMap(map);	
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


# Just this year
#d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
# Cheltenham
#d=d[(d.twp == 'CHELTENHAM')]
# 


# So many points at exact location
def myRand():
    r=random.random()*0.0001-0.00005
    return r

# Only use this is all points landing on the same spot.
d.lat=d.lat.apply(lambda x: str(float(x)+myRand()))
#d.lng=d.lng.apply(lambda x: str(float(x)+myRand()))


# 
k=d[d.parent_incident_type.str.match(r'Theft from Vehic*')]
t=k[['title','timeStamp','lat','lng','desc','twp']]
t=t[(t['timeStamp'] >= '2017-01-03 00:00:00')]
title=t.title.tolist()
desc=t.desc.tolist()
twp=t.twp.tolist()
timeStamp=t.timeStamp.tolist()
lat=t.lat.tolist()
lng=t.lng.tolist()
for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s ', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/tfvA.png'],\n" % (displayTitle,lat[i],lng[i],i)


# < 2017

k=d[d.parent_incident_type.str.match(r'Theft from Vehic*')]
t=k[['title','timeStamp','lat','lng','desc','twp']]
t=t[(t['timeStamp'] < '2017-01-03 00:00:00')]
title=t.title.tolist()
desc=t.desc.tolist()
twp=t.twp.tolist()
timeStamp=t.timeStamp.tolist()
lat=t.lat.tolist()
lng=t.lng.tolist()
for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s ', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/tfvO.png'],\n" % (displayTitle,lat[i],lng[i],i)




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

