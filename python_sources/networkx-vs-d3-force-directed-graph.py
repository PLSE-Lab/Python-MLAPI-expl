import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import networkx as n
from operator import itemgetter
import itertools

c = sqlite3.connect('../input/database.sqlite')

a = []
b = []
z = []
dic = {}
dic["Kaggle"] = 0
dic1 = 1
p = pd.read_sql("select Teams.Id as TID, TeamName, Ranking from Teams Inner Join Competitions On Teams.CompetitionId=Competitions.Id where Ranking in (1) and substr(Deadline,1,4)='2015'",c)
for i in range(len(p.TID)):
    q = pd.read_sql("select UserId from TeamMemberships where TeamId='" + str(p.TID[i])+ "'", c)
    if str(p.TeamName[i]) not in dic:
        dic[str(p.TeamName[i])] = int(dic1)
        dic1 +=1
    a.append(["Kaggle",str(p.TeamName[i]),p.Ranking[i]])
    for l in range(len(q.UserId)):
        r = pd.read_sql("select Id as UID, DisplayName from Users where Id='" + str(q.UserId[l]) + "'", c)
        for m in range(len(r.DisplayName)):
            if str(r.DisplayName[m]) not in dic:
                dic[str(r.DisplayName[m])] = int(dic1)
                dic1 +=1
            a.append([str(p.TeamName[i]),str(r. DisplayName[m]), p.Ranking[i]])
#print(a)
a= sorted(a,key=itemgetter(1))
#print(dic)
for dict1 in dic:
    b.append([str(dict1), int(dic[str(dict1)])])
b= sorted(b,key=itemgetter(1))
print(b)

# networkx Graph
g = n.Graph()
g.clear()
for i in range(len(a)):
    g.add_edge(a[i][0],a[i][1])
pos=n.spring_layout(g)
n.draw(g)
n.draw_networkx_labels(g,pos,font_size=9,font_family='sans-serif')
plt.savefig('team_plot.png')
plt.clf()

#http://bl.ocks.org/mbostock/4062045

f = open("g.json","w")
f.write("{\"nodes\":[")
str1=""
for i in range(len(b)):
    str1+="{\"name\":\""+ str(b[i][0]) + "\",\"group\":" + str(b[i][1]) +"},"
f.write(str1[:-1])
f.write("],\"links\":[")
str1=""
for i in range(len(a)):
    str1+="{\"source\":" + str(dic[str(a[i][0])]) + ",\"target\":" + str(dic[str(a[i][1])]) + ",\"value\":" + str(4-int(a[i][2])) + "},"
f.write(str1[:-1])
f.write("]}")
f.close

h1 = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
.link {stroke: #ccc;}
.node text {pointer-events: none;  font: 10px sans-serif;}
</style>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script>
var width = 800, height = 800;
var color = d3.scale.category20();
var force = d3.layout.force()
    .charge(-120)
    .linkDistance(80)
    .size([width, height]);
var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);
d3.json("g.json", function(error, graph) {
  if (error) throw error;
	force.nodes(graph.nodes)
	    .links(graph.links)
	    .start();
	var link = svg.selectAll(".link")
	    .data(graph.links)
	    .enter().append("line")
	    .attr("class", "link")
	    .style("stroke-width", function (d) {return Math.sqrt(d.value);});
	var node = svg.selectAll(".node")
	    .data(graph.nodes)
	    .enter().append("g")
	    .attr("class", "node")
	    .call(force.drag);
	node.append("circle")
	    .attr("r", 8)
	    .style("fill", function (d) {return color(d.group);})
	node.append("text")
	      .attr("dx", 10)
	      .attr("dy", ".35em")
	      .text(function(d) { return d.name });
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
</script>
"""

f = open("output.html","w")
f.write(h1)
f.close