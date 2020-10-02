#Creates a weighted, directed graph of all of the Clinton emails of the type
# email_sender ------weight-------> email_recipient
# where "email_sender" and "email_recipient" are nodes and
# weight is the weight of the edge, defined
# as the number of emails sent by email_sender to email_recipient
# Outputs an html file that displays a d3.js force directed graph

#first the imports

import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
import json
from networkx.readwrite import json_graph

# read the main data source
emails = pd.read_csv("../input/Emails.csv")

#cleanup the names in the From and To fields
with open("../input/Aliases.csv") as f:
    file = f.read().split("\r\n")[1:] #skip the header line
    aliases = {}
    for line in file:
        line = line.split(",")
        aliases[line[1]] = line[2]

with open("../input/Persons.csv") as f:
    file = f.read().split("\r\n")[1:] #skip header line
    persons = {}
    for line in file:
        line = line.split(",")
        persons[line[0]] = line[1]
        
def resolve_person(name):
    name = str(name).lower().replace(",","").split("@")[0]
    #print(name)
    #correct for some of the common people who are resolved to several different
    # names by the given Aliases.csv file:  Cheryl Mills, Huma Abedin, Jake Sullivan
    # and Lauren Jiloty
    # Also convert "h" and variations to Hillary Clinton
    if ("mills" in name) or ("cheryl" in name) or ("nill" in name) or ("miliscd" in name) or ("cdm" in name) or ("aliil" in name) or ("miliscd" in name):
        return "Cheryl Mills"
    elif ("a bed" in name) or ("abed" in name) or ("hume abed" in name) or ("huma" in name) or ("eabed" in name):
        return "Huma Abedin"
    #elif (name == "abedin huma") or (name=="huma abedin") or (name=="abedinh"): 
    #    return "Huma Abedin"
    elif ("sullivan" in name)  or ("sulliv" in name) or ("sulliy" in name) or ("su ii" in name) or ("suili" in name):
        return "Jake Sullivan"
    elif ("iloty" in name) or ("illoty" in name) or ("jilot" in name):
        return "Lauren Jiloty"
    elif "reines" in name: return "Phillip Reines"
    elif (name == "h") or (name == "h2") or ("secretary" in name) or ("hillary" in name) or ("hrod" in name):
        return "Hillary Clinton"
    #fall back to the aliases file
    elif str(name) == "nan": return "Redacted"
    elif name in aliases.keys():
        return persons[aliases[name]]
    else: return name
    
emails.MetadataFrom = emails.MetadataFrom.apply(resolve_person)
emails.MetadataTo = emails.MetadataTo.apply(resolve_person)

#Extract the to: from: and Raw body text from each record

From_To_RawText = []
temp = zip(emails.MetadataFrom,emails.MetadataTo,emails.RawText)
for row in temp:
    From_To_RawText.append(((row[0],row[1]),row[2]))

#Create a dictionary of all edges, i.e. (sender, recipient) relationships 

From_To_allText = defaultdict(list)
for people, text in From_To_RawText:
    From_To_allText[people].append(text)
len(From_To_allText.keys()), len(From_To_RawText)

#Set the weights of each directed edge equal to the number of emails 
# (number of raw text documents) associated with that edge
edges_weights = [[key[0], key[1], len(val)] for key, val in From_To_allText.items()]
edge_text = [val for key, val in From_To_allText.items()]

#initialize the graph
graph = nx.DiGraph()

#transform the dict with keys (from,to) and vals weight back to a 
# tuple(from, to, weight)
graph.add_weighted_edges_from(edges_weights)
#nx.set_edge_attributes(graph, 'text', edge_text)

#Calculate the pagerank of each person (node) and store it with the node.
pagerank = nx.pagerank(graph)
pagerank_list = {node: rank for node, rank in pagerank.items()}
nx.set_node_attributes(graph, 'pagerank', pagerank_list)


#save the graph to json format to be read by d3.js
graph_json = json.dumps(json_graph.node_link_data(graph))
with open("clintongraph.json",'w') as f:
    f.write(graph_json)
    
    
#This is the html file to display the force directed graph
html = '''
<!DOCTYPE html>
<meta charset="utf-8">

<body>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<!DOCTYPE html>
<meta charset="utf-8">
<style>
.link {
  stroke: #ccc;
}

.linktext text {
	font: 5px sans-serif;
	stroke: gray;
	text-anchor: middle;
}

.node text {
  pointer-events: all;
  font: 6px sans-serif;
  stroke: black;
  fill: black;
  text-anchor: middle;
}


</style>
<body>

<script>

// main script text from Mike Bostock http://bl.ocks.org/mbostock/4062045
// additional code from http://www.coppelia.io/2014/07/an-a-to-z-of-extra-features-for-the-d3-force-layout/

var currentWidth;
currentWidth = function() {
	return [window.innerWidth, window.innerHeight];
}

var width = window.innerWidth,
    height = window.innerHeight;
var color = d3.scale.category10();
var force = d3.layout.force()
    .charge(-150)
    .linkDistance(100)
	.gravity(0.05)
    .size([width, height]);
var svg = d3.select("body").append("svg")
    .attr("width", window.innerWidth)
    .attr("height", window.innerHeight);


d3.json("clintongraph.json", function(error, graph) {
  if (error) throw error;
	force.nodes(graph.nodes)
	    .links(graph.links)
	    .start();
	var link = svg.selectAll(".link")
	    .data(graph.links)
	    .enter().append("line")
	    .attr("class", "link")
		.style("stroke-width", function(d) {return Math.log(d.weight)/2+.5;})
		.style("marker-end",  "url(#suit)"); // Modified line ;


	var node = svg.selectAll(".node")
	    .data(graph.nodes)
	    .enter().append("g")
	    .attr("class", "node")
	    .call(force.drag)
		.on('dblclick', connectedNodes); 
	node.append("circle")
	    .attr("r", function(d) {return 3*Math.log(d.pagerank);})
	    .style("fill", function(d) {return color(Math.log(d.pagerank));})
		.style("opacity", 0.6)
		.append("title");
	node.append("text")
	      .attr("dx", 0)
	      .attr("dy", ".35em")
		  .attr("stroke", "black")
	      .text(function(d) { return d.id });
		  

//Toggle stores whether the highlighting is on
var toggle = 0;
//Create an array logging what is connected to what
var linkedByIndex = {};
for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
};
graph.links.forEach(function (d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
});
//This function looks up whether a pair are neighbours
function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
}
function connectedNodes() {
    if (toggle == 0) {
        //Reduce the opacity of all but the neighbouring nodes
        d = d3.select(this).node().__data__;
        node.style("opacity", function (o) {
            return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
        });
        link.style("opacity", function (o) {
            return d.index==o.source.index | d.index==o.target.index ? 1 : 0.1;
        });
        //Reduce the op
        toggle = 1;
    } else {
        //Put them back to opacity=1
        node.style("opacity", 1);
        link.style("opacity", 1);
        toggle = 0;
    }
}

	force.on("tick", function () {
	    link.attr("x1", function (d) {return d.source.x;})
		.attr("y1", function (d) {return d.source.y;})
		.attr("x2", function (d) {return d.target.x;})
		.attr("y2", function (d) {return d.target.y;});
	    d3.selectAll("circle").attr("cx", function (d) {return d.x;})
		.attr("cy", function (d) {return d.y;});
	    d3.selectAll("text").attr("x", function (d) {return d.x;})
		.attr("y", function (d) {return d.y;});
  });
});

svg.append("defs").selectAll("marker")
    .data(["suit", "licensing", "resolved"])
  .enter().append("marker")
    .attr("id", function(d) { return d; })
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 25)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("path")
    .attr("d", "M0,-5L10,0L0,5 L10,0 L0, -5")
    .style("stroke", "#4679BD")
    .style("opacity", "0.6");
</script>
'''

with open("clintongraph.html","w") as f:
    f.write(html)