"""Author-Khwaja Wisal Maqsood"""
 """  This script is leveraging two Kernels.  There's an 
   embedded graph in the map, currently off to the left
   which plots helicopter landings.  That graphic
   was created in the following separate kernel
   https://www.kaggle.com/khwajawisal/graphic-embedding-of-maps/edit
      




"""


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':datetime.datetime,'twp':str,'addr':str,'e':int}, 
     )

# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
#d=d[(d.timeStamp >= "2016-01-01 00:00:00")]

import pandas as pd
import numpy as np
import datetime


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
  <body> <!--  DataCanary_s fix -->
      <div id="map" class="main-container"></div>
    <script>

      function initMap() {
      var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 10,
      center: {lat: 40.063168, lng: -75.149746}
     
      });

      setMarkers(map);
      
      // This is for the graph in the map
      
      
       var imageBounds = {
         north: 39.945713,
         south: 39.822263,
         east:  -75.679667,
         west:  -75.881887
      };


       graphOverlay = new google.maps.GroundOverlay(
         'https://www.kaggle.io/svf/335955/5dbcf966e98eb8bd5b0c470c3eeb9445/t120helicopter.svg',
         imageBounds);
      graphOverlay.setMap(map);

      
      
      
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


# Select

# Just this year
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]



k=d[d.title.str.match(r'.*HELICOPTER.*')]
t=k[['title','timeStamp','lat','lng','desc','twp']]
title=t.title.tolist()
desc=t.desc.tolist()
twp=t.twp.tolist()
timeStamp=t.timeStamp.tolist()
lat=t.lat.tolist()
lng=t.lng.tolist()
for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s ', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/helicopter.png'],\n" % (displayTitle,lat[i],lng[i],i)


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


