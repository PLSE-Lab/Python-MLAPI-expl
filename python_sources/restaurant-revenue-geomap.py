output = '''
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>GeoMapID475257fe3d32</title>
<meta http-equiv="content-type" content="text/html;charset=utf-8" />
<style type="text/css">
body {
  color: #444444;
  font-family: Arial,Helvetica,sans-serif;
  font-size: 75%;
  }
  a {
  color: #4D87C7;
  text-decoration: none;
}
</style>
</head>
<body>
 <!-- GeoMap generated in R 3.1.1 by googleVis 0.5.8 package -->
<!-- Fri May  8 10:27:46 2015 -->


<!-- jsHeader -->
<script type="text/javascript">
 
// jsData 
function gvisDataGeoMapID475257fe3d32 () {
var data = new google.visualization.DataTable();
var datajson =
[
 [
 "Adana",
4894944.333,
"Restaurants: 3" 
],
[
 "Afyonkarahisar",
4952497,
"Restaurants: 1" 
],
[
 "Amasya",
2954086,
"Restaurants: 1" 
],
[
 "Ankara",
3275405.842,
"Restaurants: 19" 
],
[
 "Antalya",
3481448.25,
"Restaurants: 4" 
],
[
 "Aydn",
3429798.5,
"Restaurants: 2" 
],
[
 "Balkesir",
4758476,
"Restaurants: 1" 
],
[
 "Bolu",
4263629,
"Restaurants: 1" 
],
[
 "Bursa",
4092295,
"Restaurants: 5" 
],
[
 "Denizli",
2344689,
"Restaurants: 1" 
],
[
 "Diyarbakr",
3735351.333,
"Restaurants: 3" 
],
[
 "Edirne",
5444227,
"Restaurants: 1" 
],
[
 "Elaz",
5525735,
"Restaurants: 1" 
],
[
 "Eskiehir",
3957953.333,
"Restaurants: 3" 
],
[
 "Gaziantep",
4316715,
"Restaurants: 1" 
],
[
 "Isparta",
4015749,
"Restaurants: 1" 
],
[
 "stanbul",
5577811.96,
"Restaurants: 50" 
],
[
 "zmir",
5287570.778,
"Restaurants: 9" 
],
[
 "Karabk",
3807496,
"Restaurants: 1" 
],
[
 "Kastamonu",
3273041,
"Restaurants: 1" 
],
[
 "Kayseri",
4567575.667,
"Restaurants: 3" 
],
[
 "Krklareli",
1619683,
"Restaurants: 1" 
],
[
 "Kocaeli",
3745135,
"Restaurants: 1" 
],
[
 "Konya",
2667256.5,
"Restaurants: 2" 
],
[
 "Ktahya",
2993069,
"Restaurants: 1" 
],
[
 "Mula",
4111129,
"Restaurants: 2" 
],
[
 "Osmaniye",
3376145,
"Restaurants: 1" 
],
[
 "Sakarya",
3328853.25,
"Restaurants: 4" 
],
[
 "Samsun",
3247869.6,
"Restaurants: 5" 
],
[
 "anlurfa",
3261924,
"Restaurants: 1" 
],
[
 "Tekirda",
3312470.667,
"Restaurants: 3" 
],
[
 "Tokat",
2675511,
"Restaurants: 1" 
],
[
 "Trabzon",
5284100.5,
"Restaurants: 2" 
],
[
 "Uak",
1763231,
"Restaurants: 1" 
] 
];
data.addColumn('string','City');
data.addColumn('number','Average.Revenue');
data.addColumn('string','Restaurants');
data.addRows(datajson);
return(data);
}
 
// jsDrawChart
function drawChartGeoMapID475257fe3d32() {
var data = gvisDataGeoMapID475257fe3d32();
var options = {};
options["dataMode"] = "regions";
options["width"] =    556;
options["height"] =    350;
options["region"] = "TR";


    var chart = new google.visualization.GeoMap(
    document.getElementById('GeoMapID475257fe3d32')
    );
    chart.draw(data,options);
    

}
  
 
// jsDisplayChart
(function() {
var pkgs = window.__gvisPackages = window.__gvisPackages || [];
var callbacks = window.__gvisCallbacks = window.__gvisCallbacks || [];
var chartid = "geomap";
  
// Manually see if chartid is in pkgs (not all browsers support Array.indexOf)
var i, newPackage = true;
for (i = 0; newPackage && i < pkgs.length; i++) {
if (pkgs[i] === chartid)
newPackage = false;
}
if (newPackage)
  pkgs.push(chartid);
  
// Add the drawChart function to the global list of callbacks
callbacks.push(drawChartGeoMapID475257fe3d32);
})();
function displayChartGeoMapID475257fe3d32() {
  var pkgs = window.__gvisPackages = window.__gvisPackages || [];
  var callbacks = window.__gvisCallbacks = window.__gvisCallbacks || [];
  window.clearTimeout(window.__gvisLoad);
  // The timeout is set to 100 because otherwise the container div we are
  // targeting might not be part of the document yet
  window.__gvisLoad = setTimeout(function() {
  var pkgCount = pkgs.length;
  google.load("visualization", "1", { packages:pkgs, callback: function() {
  if (pkgCount != pkgs.length) {
  // Race condition where another setTimeout call snuck in after us; if
  // that call added a package, we must not shift its callback
  return;
}
while (callbacks.length > 0)
callbacks.shift()();
} });
}, 100);
}
 
// jsFooter
</script>
 
<!-- jsChart -->  
<script type="text/javascript" src="https://www.google.com/jsapi?callback=displayChartGeoMapID475257fe3d32"></script>
 
<!-- divChart -->
  
<div id="GeoMapID475257fe3d32" 
  style="width: 556; height: 350;">
</div>
 <div><span>Data: TFI &#8226; Chart ID: <a href="Chart_GeoMapID475257fe3d32.html">GeoMapID475257fe3d32</a> &#8226; <a href="https://github.com/mages/googleVis">googleVis-0.5.8</a></span><br /> 
<!-- htmlFooter -->
<span> 
  R version 3.1.1 (2014-07-10) 
  &#8226; <a href="https://developers.google.com/terms/">Google Terms of Use</a> &#8226; <a href="https://google-developers.appspot.com/chart/interactive/docs/gallery/geomap">Documentation and Data Policy</a>
</span></div>
</body>
</html>
'''

with open("output.html", "w") as output_file:
    output_file.write(output)