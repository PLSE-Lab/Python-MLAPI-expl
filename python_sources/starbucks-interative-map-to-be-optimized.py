import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import geojson as geojson


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


df = pd.read_csv("../input/directory.csv")
df.head(1)

df = df[df['Longitude'].notnull()]
df = df[df['Latitude'].notnull()]
df= df.fillna('00')

stores = zip(df['Longitude'], df['Latitude'], df['Store Number'],df['Store Name'],df['Ownership Type'],df['City'],df['State/Province'],df['Country'])
points = [geojson.Feature(geometry=geojson.Point((v[0], v[1])), properties={'Store Number': v[2],'Store Name': v[3],'Ownership Type': v[4],'City': v[5],'State/Province': v[6],'Country': v[7]}) for v in stores]

geo_collection = geojson.FeatureCollection(points)

dump = geojson.dumps(geo_collection, sort_keys=True)

with open('Stores.geojson', 'w') as file:
    file.write(dump)

headHTML= """<!DOCTYPE html>
<html>
  <head>
    <title>Starbucks Stores</title>
    <meta name="viewport" content="initial-scale=1.0">
    <meta charset="utf-8">
 
      <script type="text/javascript" src="https://maps.googleapis.com/maps/api/js?v=3&senseo=false&ext=.js"></script>
    
      <script type="text/javascript" src="http://google-maps-utility-library-v3.googlecode.com/svn/trunk/markerclustererplus/src/markerclusterer.js"></script>
    
      <script type="text/javascript" src="https://cdn.rawgit.com/googlemaps/v3-utility-library/master/infobox/src/infobox.js"></script>
    
      <script type="text/javascript" src="https://cdn.rawgit.com/googlemaps/v3-utility-library/master/markerclustererplus/src/markerclusterer.js"></script>

    <style>
html {
	height: 100%;
}
body {
	height: 100%;
	margin: 0;
	padding: 0;
}
#map {
    height: 100%;
}
.gm-style-iw {
	width: 350px !important;
	top: 15px !important;
	left: 0px !important;
	background-color: #fff;

}
#iw-container {
	margin-bottom: 10px;
	width: 100%;
}
#iw-container .iw-title {
	width: 300px !important;
	font-family: 'Open Sans Condensed', sans-serif;
	font-size: 22px;
	font-weight: 400;
	padding: 10px;
	background-color: #067655;
	color: white;
	margin: 0;
	border-radius: 2px 2px 0 0;
}
#iw-container .iw-content {
	font-size: 13px;
	line-height: 18px;
	font-weight: 400;
	margin-right: 1px;
	padding: 15px 5px 20px 15px;
	max-height: 140px;
	overflow-y: auto;
	overflow-x: hidden;
	background-color: #ffffff;
}
.iw-subTitle {
	font-size: 16px;
	font-weight: 700;
	padding: 5px 0;
}

    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>



var map = null;
var bounds = new google.maps.LatLngBounds();
var markerClusterer = new MarkerClusterer(null,null,{imagePath: "https://cdn.rawgit.com/googlemaps/v3-utility-library/master/markerclustererplus/images/m"});

  var container = '<div id="iw-container">'
  var div = '</div>'
  var content = '<div class="iw-content">'

   var infowindow =new InfoBox();
   var pinImage = new google.maps.MarkerImage('http://i.imgur.com/dRGP7Gw.png',
        new google.maps.Size(21, 34),
        new google.maps.Point(0,0),
        new google.maps.Point(10, 34));




function initialize() {
    var mapOptions = {
        center: new google.maps.LatLng(52, 8),
        zoom: 4
    };
    map = new google.maps.Map(document.getElementById('map'), mapOptions);

    markerClusterer.setMap(map);
    google.maps.event.addListener(map.data, 'addfeature', function (e) {
        if (e.feature.getGeometry().getType() === 'Point') {
            var marker = new google.maps.Marker({
                position: e.feature.getGeometry().get(),
                title: e.feature.getProperty('name'),
                icon: pinImage,
                map: map
            });
            google.maps.event.addListener(marker, 'click', function (marker, e) {
                return function () {
                    var name = '<div class="iw-title">'+e.feature.getProperty("Store Name")+div;
                    var city = '<div class="iw-subTitle"> City: '+e.feature.getProperty("City")+div;
                    var country = '<div class="iw-subTitle">Country: '+e.feature.getProperty("Country")+div;
                    var OwnerType = '<div class="iw-subTitle">Ownership Type: '+e.feature.getProperty("Ownership Type")+div;
                    var storenumber ='<div class="iw-subTitle">Store Number: '+e.feature.getProperty("Store Number")+div;
                     var StateProvince ='<div class="iw-subTitle">State/Province: '+e.feature.getProperty("State/Province")+div;

      
                     var infow = container+name+content+city+country+OwnerType+storenumber+StateProvince+div+div
 

                      infowindow.setPosition(e.feature.getGeometry().get());
                      infowindow.setOptions({pixelOffset: new google.maps.Size(0,-30)});
                      infowindow.setContent(infow);
                      infowindow.open(map);
      
  };}(marker, e));
            markerClusterer.addMarker(marker);
            bounds.extend(e.feature.getGeometry().get());
            map.fitBounds(bounds);
            map.setCenter(e.feature.getGeometry().get());
        }
    });
    layer = map.data.loadGeoJson('Stores.geojson');
    map.data.setMap(null);

}
google.maps.event.addDomListener(window, 'load', initialize);
    </script>

  </body>
</html> """


f=open('__results__.html','w')
f.write(headHTML)

f.close()

# Write out 
f=open('output.html','w')
f.write(headHTML)

f.close()

import codecs
f=codecs.open("output.html", 'r')
print (f.read())