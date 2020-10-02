# This was taken from the following:
#   https://www.kaggle.com/mchirico/d/mchirico/philadelphiacrimedata/burglary-residential-2016
#   


import numpy as np 
import pandas as pd 
import seaborn as sns

d = pd.read_csv('../input/crime_homicide_subset.csv', encoding='latin1', sep=',')



print(d.head())

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
      zoom: 14,
      center: {lat: 38.919302, lng: -77.034831}
      // Washington
     // 38.919302, -77.034831
      });

      setMarkers(map);
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


# mike, change this variable...please!
s=' var crashes = [\n'


# Select

d=d[(d['OFFENSE'] == 'HOMICIDE') &  (d['METHOD'] == 'GUN')]
d.fillna(0, inplace=True)
d=d[(d.long != 0 ) | (d.lat != 0 )]

# Just get the headers that you want.

k=d
t=k[['OFFENSE','REPORT_DAT','long','lat','VOTING_PRECINCT','hour']]

# Change these mike...to be meaningful, after working
title=t.OFFENSE.tolist()
desc=t.VOTING_PRECINCT.tolist()
twp=t.hour.tolist()
timeStamp=t.REPORT_DAT.tolist()
lat=t.lat.tolist()
lng=t.long.tolist()



for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/shooting.png'],\n" % (displayTitle,lat[i],lng[i],i)


s+='];'

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
