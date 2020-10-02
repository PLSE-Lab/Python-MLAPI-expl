#!/usr/bin/env python
# coding: utf-8

# # [Visualization Notes](https://visualization-notes.blogspot.com)
# ## Additional Packages: Pygal

# In[ ]:


get_ipython().system('python3 -m pip install pygal --user')


# In[ ]:


import os
os.listdir('../working/')


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
from IPython.display import display,HTML
import pygal,pandas as pd
from pygal.style import BlueStyle
columns=['Fresh','Milk','Grocery','Frozen',
         'Detergents_Paper','Delicatessen']
colors=['#1b2c45','#5a8bbd','#008b8b','#ff5a8b']
index= ['C1','C2','C3']
data=[[26373,36423,22019,5154,4337,16523],
      [16165,4230,7595,201,4003,57],
      [14276,803,3045,485,100,518]]
samples=pd.DataFrame(data,columns=columns,index=index) 
line=pygal.Line(fill=False,height=500,
                style=BlueStyle(opacity='.3',colors=colors,
                                background='transparent'))
line.title='Samples of the Dataset "Wholesale Customers"'
line.x_labels=columns
line.add('C1',data[0]); line.add('C2',data[1])
line.add('C3',data[2]); line.add('MEAN',samples.mean())
line.render_to_file('samples.svg')
s1='<figure><embed type="image/svg+xml" '
s2='src="samples.svg" width=500/></figure>'
HTML(s1+s2)


# ## Interactive Outputs: D3 Mouse Events & Others

# In[ ]:


html_str='''
<style>
svg {background:silver;}
text {fill:#ff355e;}
rect {fill:none; pointer-events:all;} 
circle {fill:none; stroke-width:2px;}
</style>
<div id="d3viz"></div>
<script src="https://d3js.org/d3.v5.min.js"></script><script>
var width=500, height=300, i=0;
var svg=d3.select("#d3viz").append("svg")
          .attr("width",width).attr("height",height);
var r=svg.append("rect")
         .attr("width",width).attr("height",height)
         .on("ontouchstart" in document ? 
             "touchmove":"mousemove",draw);
function draw() {
  var m=d3.mouse(this);
  svg.insert("circle","rect")
     .attr("cx",m[0]).attr("cy",m[1]).attr("r",1e-6)
     .style("stroke",d3.hsl((i=(i+1)%360),1,.5))
     .style("stroke-opacity",1)
     .transition().duration(2000)
     .ease(Math.sqrt).attr("r",10)
     .style("stroke-opacity",1e-6)
     .remove();
  d3.event.preventDefault();};
  svg.append("text").attr("x",25).attr("y",25)
     .text("D3 Mouse Events");
</script>'''
html_file=open("d3.html","w")
html_file.write(html_str); html_file.close()
HTML('''<div id='data1'><iframe src="d3.html" 
height="320" width="520"></iframe></div>''')


# In[ ]:


html_str='''
<script src='//d3js.org/d3.v3.min.js'></script>
<svg id='poly' style='background-color:silver;'></svg>
<script>
var mouse=[330,330],count=0;
var svg=d3.select('#poly')
          .attr('width',500).attr('height',500);
var g=svg.selectAll('g').data(d3.range(25)).enter()
         .append('g').attr('transform','translate('+mouse+')');
g.append('rect').attr('rx',5).attr('ry',5)
 .attr('x',-5).attr('y',-5)
 .attr('width',20).attr('height',20)
 .attr('transform',function(d,i){return 'scale('+(1-d/25)*20+')';})
 .style('fill',d3.scale.category20b());
g.datum(function(d){return {center:mouse.slice(),angle:0};});
svg.on('mousemove',function(){mouse=d3.mouse(this);});
d3.timer(function(){count++; 
g.attr('transform',
       function(d,i){d.center[0]+=(mouse[0]-d.center[0])/(i+1); 
d.center[1]+=(mouse[1]-d.center[1])/(i+1); 
d.angle+=Math.sin((count+i)/40)*7; 
return 'translate('+d.center+')rotate('+d.angle+')'});});
</script>'''
fpath='../kaggle-static/static/dist/jupyterlab/'
html_file=open("d3_2.html","w")
html_file.write(html_str); html_file.close()
HTML('''<div id='data2'>
<iframe src="d3_2.html" 
height="520" width="520"></iframe></div>''')


# ## Interactive Code Cells 

# In[ ]:


get_ipython().run_cell_magic('html', '', '<div style="border:10px double white; \n     width:550px; height:950px; overflow:auto; \n     padding:10px; background-color:ghostwhite">\n<iframe src="https://olgabelitskaya.github.io/kaggle_smc.html" \n        width="510" height="920"/></div>')

