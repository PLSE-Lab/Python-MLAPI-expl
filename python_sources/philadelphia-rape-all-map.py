# This script/Kernel will create a Google Map 
# 


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

FILE="../input/crime.csv"

d=pd.read_csv(FILE,
  header=0,names=['Dc_Dist', 'Psa', 'Dispatch_Date_Time', 'Dispatch_Date',
       'Dispatch_Time', 'Hour', 'Dc_Key', 'Location_Block', 'UCR_General',
       'Text_General_Code',  'Police_Districts', 'Month', 'Lon',
       'Lat'],dtype={'Dc_Dist':str,'Psa':str,
                'Dispatch_Date_Time':str,'Dispatch_Date':str,'Dispatch_Time':str,
                  'Hour':str,'Dc_Key':str,'Location_Block':str,
                     'UCR_General':str,'Text_General_Code':str,
              'Police_Districts':str,'Month':str,'Lon':str,'Lat':str},
             parse_dates=['Dispatch_Date_Time'],date_parser=dateparse)





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
      zoom: 13,
      center: {lat: 40.005326, lng: -75.154777}
     
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

# Set index

# Set index
d.index = pd.DatetimeIndex(d.Dispatch_Date_Time	)
# Just this year
#d=d[(d.Dispatch_Date_Time	 >= "2012-01-01 00:00:00")]

# mike, change this variable...please!
s=' var crashes = [\n'


# Select

d.index = pd.DatetimeIndex(d.Dispatch_Date_Time	)
# Just this year
#d=d[(d.Dispatch_Date_Time	 >= "2016-01-01 00:00:00")]

# Missing some lat/lon
# If nan's get on the graph, it won't display.
d.fillna(0, inplace=True)
d=d[(d.Lon != 0 ) | (d.Lat != 0 )]


k=d[(d.Text_General_Code == 'Rape')]
t=k[['Text_General_Code','Dispatch_Date_Time','Lon','Lat','Location_Block','Hour']]

# Change these mike...to be meaningful, after working
title=t.Text_General_Code.tolist()
desc=t.Location_Block.tolist()
twp=t.Hour.tolist()
timeStamp=t.Dispatch_Date_Time.tolist()
lat=t.Lat.tolist()
lng=t.Lon.tolist()



for i in range(0,len(lat)):
    displayTitle="%s %s %s %s" % (title[i],desc[i],twp[i],timeStamp[i])
    displayTitle=displayTitle.replace('\n',' ')
    s+="['%s', %s, %s, %s,'https://storage.googleapis.com/montco-stats/images/rape.png'],\n" % (displayTitle,lat[i],lng[i],i)


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








