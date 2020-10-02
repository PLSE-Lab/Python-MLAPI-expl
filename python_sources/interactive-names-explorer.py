import pandas as pd
import json
import numpy as np

datapath = '../input/NationalNames.csv'
dn = pd.read_csv(datapath,index_col='Id')
dn['CountYear'] = dn.groupby(['Year','Gender'])['Count'].transform('sum')
dn['Popularity'] = np.round(1000*dn.Count.values / dn.CountYear.values,2) #babies per thousand
df = dn[dn.Popularity >=1.]
jsondata = {}
for g, ggroup in df.groupby(df.Gender):
    gdata = {}
    for y, ygroup in ggroup.groupby(ggroup.Year):
        yeardata = {'children':[]}
        for n, ngroup in ygroup.groupby(ygroup.Name.str[0]):
            gr = {'name':n, 'popularity':ngroup.Popularity.sum(),'children':[]}
            for row in ngroup.itertuples():
                gr['children'].append({'name':row.Name, 'popularity':row.Popularity})
            yeardata['children'].append(gr)
        gdata['Y{}'.format(y)] = yeardata
    jsondata[g] = gdata
with open('namesdata.json', 'w') as f:
    json.dump(jsondata, f)

html = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Interactive Names Explorer</title>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/4.2.6/d3.min.js"></script>
<style type='text/css'>
    body {font-family:Arial, sans-serif;}
    span.gender {font-family:Arial; font-size:15px;padding:8px;border-radius:6px;color:#444}
    span.gender:hover{color:#000;}
    span.selected {background-color: #eee;border: 1px solid #ddd;color:#000;}
    .ticks {
      font: 10px sans-serif;
    }
    .track,
    .track-inset,
    .track-overlay {
      stroke-linecap: round;
    }
    .track {
      stroke: #000;
      stroke-opacity: 0.3;
      stroke-width: 10px;
    }
    .track-inset {
      stroke: #ddd;
      stroke-width: 8px;
    }
    .track-overlay {
      pointer-events: stroke;
      stroke-width: 50px;
      cursor: crosshair;
    }
    .handle {
      fill: #fff;
      stroke: #000;
      stroke-opacity: 0.5;
      stroke-width: 1.25px;
    }
</style>
</head>
<body>
    <div id='gender'>
        <span id='girls' class='gender selected'>Girl names</span>
        <span id='boys' class='gender'>Boy names</span>
    </div>
    <div id='slider'></div>
    <div id="chart"> </div>
    <div id='scale'></div>
<script type="text/javascript">
var w = 600;
var h = 600;
var sliderHeight = 80;
var sliderMargin = 40;
var buttonRadius = 12;
var scaleHeight = 40;
var padding = 1;
var stretch = 3;
var transitionDuration = 500;
var currentYear = '2014';
var currentGender = 'F';
var svg = d3.select("#chart")
          .append("svg")
          .attr("width",w)
          .attr("height",h);
var lettercolors = {};
for (var i=0;i<26;i++) {
    lettercolors["ABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(i)] = d3.interpolateRainbow(Math.min(25,i+1)/25.);
}
var key = function(d) { return d.data.name; }
var treemap = d3.treemap()
             .size([w/stretch,h])
             .round(true)
// Gender selector
var girls = d3.select('#girls');
var boys = d3.select('#boys');
girls.on('click',function() {girls.attr('class','gender selected');
                             boys.attr('class','gender');
                             update_chart('F', currentYear);})
boys.on('click',function() {boys.attr('class','gender selected');
                             girls.attr('class','gender');
                             update_chart('M', currentYear);})
// Create years slider
var x = d3.scaleLinear()
    .domain([1880, 2014])
    .rangeRound([0, w-2*sliderMargin])
    .clamp(true);

var slider = d3.select('#slider')
    .append('svg')
    .attr('width',w)
    .attr('height',sliderHeight)
    .append("g")
    .attr("class", "slider")
    .attr("transform", "translate(" +sliderMargin+ "," + sliderHeight / 2 + ")");

slider.append("line")
    .attr("class", "track")
    .attr("x1", x.range()[0])
    .attr("x2", x.range()[1])
    .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-inset")
    .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-overlay")
    .call(d3.drag()
    .on("start.interrupt", function() { slider.interrupt(); })
    .on("end drag", function() { update_chart(currentGender, Math.round(x.invert(d3.event.x))); }));

slider.insert("g", ".track-overlay")
    .attr("class", "ticks")
    .attr("transform", "translate(0," + 18 + ")")
    .selectAll("text")
    .data(x.ticks(10))
    .enter().append("text")
    .attr("x", x)
    .attr("text-anchor", "middle")
    .text(function(d) { return d; });

var handle = slider.insert("circle", ".track-overlay")
    .attr("class", "handle")
    .attr("r", 9);
var handlelabel = slider.insert('text','.track-overlay')
    .attr('class','handlelabel')
    .text('2014')
    .attr("x", x.range()[1])
    .attr("y", -15)
    .attr('text-anchor','middle')
var yearup = d3.select('#slider svg').append("circle")
    .attr("class", "handle")
    .attr("r", 12)
    .attr('cx',w-buttonRadius-2)
    .attr('cy',sliderHeight/2)
    .on('click', function() {currentYear = Math.min(2014, currentYear+1); update_chart(currentGender, currentYear)});
var yearuplabel = d3.select('#slider svg').append("text")
    .text(">")
    .attr('x',w-buttonRadius-2)
    .attr('text-anchor','middle')
    .attr('dy','0.35em')
    .attr('y',sliderHeight/2)
    .style('pointer-events','none');
var yeardown = d3.select('#slider svg').append("circle")
    .attr("class", "handle")
    .attr("r", 12)
    .attr('cx',buttonRadius+2)
    .attr('cy',sliderHeight/2)
    .on('click', function() {currentYear = Math.max(1880, currentYear-1); update_chart(currentGender, currentYear)});;
var yeardownlabel = d3.select('#slider svg').append("text")
    //.attr("class", "handle")
    .text("<")
    .attr('x',buttonRadius+2)
    .attr('text-anchor','middle')
    .attr('dy','0.35em')
    .attr('y',sliderHeight/2)
    .style('pointer-events','none');
// Create area scale
var scale = d3.select('#scale')
    .append('svg')
    .attr("width",w)
    .attr("height",scaleHeight);
var scaleRect = scale.append('rect')
    .attr('width',scaleHeight-10)
    .attr('height',scaleHeight-10)
    .attr('x',5)
    .attr('y',5)
    .attr('fill',lettercolors['A'])
var scaleText = scale.append('text')
    .text('One baby per thousand')
    .attr('x',scaleHeight+5)
    .attr('y',scaleHeight/2).attr('dy','0.35em')

var update_chart = function(gender, year) {
    currentYear = year;
    currentGender = gender;
    handle.attr("cx", x(year));
    handlelabel.text(year).attr('x',x(year));
    var root = d3.hierarchy(dataset[gender]['Y'+year])
    root.sum(function(d) { return d.children ? 0 : d.popularity; });
    treemap(root);
    var node = svg.selectAll("rect").data(root.leaves(), key);
    var labels = svg.selectAll("text").data(root.leaves(), key);
    // Adjust scale
    scalew = Math.round(Math.sqrt(w*h/root.value));
    scaleRect.attr('width',scalew)
    .attr('height',scalew)
    .attr('x',Math.round((scaleHeight-scalew)/2))
    .attr('y',Math.round((scaleHeight-scalew)/2))
    //Add
    node.enter()
        .append('rect')
        .attr('x',function(d) {return (d.x0+d.x1)*stretch/2;})
        .attr('y',function(d) {return (d.y0+d.y1)/2+5;})
        .attr('width', 0)
        .attr('height',0)
        .attr('title', function(d) {return d.data.name + " " + d.data.popularity.toFixed(2);})
        .transition()
        .duration(transitionDuration)
        .attr("x", function(d) {return d.x0*stretch;})
        .attr("y", function(d) {return d.y0;})
        .attr('width', function(d) {return (d.x1-d.x0)*stretch-1;})
        .attr('height', function(d) {return d.y1-d.y0-1;})
        .attr('fill',function(d) {return lettercolors[d.data.name.charAt(0)];})
    labels.enter()
            .append('text')
            .attr('title', function(d) {return d.data.name + " " + d.data.popularity.toFixed(2);})
            .text(function(d) { return d.data.name;})
            .attr('x',function(d) {return (d.x0+d.x1)*stretch/2;})
            .attr('y',function(d) {return (d.y0+d.y1)/2})
            .attr('text-anchor','middle')
            .attr('font-size',function(d) {return Math.round(10+0.6*d.data.popularity) + 'px';})
            .attr('dy','0.35em');
    //Update
    node.transition()
        .duration(transitionDuration)
        .attr("x", function(d) {return d.x0*stretch;})
        .attr('width', function(d) {return (d.x1-d.x0)*stretch-1;})
        .attr("y", function(d) {return d.y0;})
        .attr('height', function(d) {return d.y1-d.y0-1;})
        .attr('title', function(d) {return d.data.name + " " + d.data.popularity;})
    labels.transition()
            .duration(transitionDuration)
            .attr('x',function(d) {return (d.x0+d.x1)*stretch/2;})
            .attr('y',function(d) {return (d.y0+d.y1)/2;})
            .attr('title', function(d) {return d.data.name + " " + d.data.popularity.toFixed(2);})
            .attr('font-size',function(d) {return Math.round(10+0.6*d.data.popularity) + 'px';});
    //Remove
    node.exit()
        .transition()
        .duration(transitionDuration)
        .attr('x',function(d) {return (d.x0+d.x1)*4/2;})
        .attr('y',function(d) {return (d.y0+d.y1)/2;})
        .attr('width', 0)
        .attr('height',0)
        .remove();
    labels.exit()
          .remove();
};
d3.json("namesdata.json", function(error, data) {
    if (error) {return console.error(error);}
  dataset = data;
  update_chart(currentGender, currentYear);
});
</script>
</body>
</html>
'''

with open('output.html','w') as f:
    f.write(html)
