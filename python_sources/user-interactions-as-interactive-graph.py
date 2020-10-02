#!/usr/bin/env python
# coding: utf-8

# In this notebook I look at some basic statistics on a graph of user interactions for high scoring questions (score > 75).  The resulting graph is rendered in D3 and allows interactive exploration of the network.
# 
# See my blog for a detailed explanation of the approach <a href='https://aster-community.teradata.com/community/learn-aster/aster-works/custom-visualizations-using-d3-and-jupyter'>here</a>.

# In[ ]:


import os
import pandas as pd
import networkx as nx
import numpy as np
import json
from IPython.display import Javascript


# In[ ]:


answers = pd.read_csv('../input/Answers.csv', encoding='latin-1')
questions = pd.read_csv('../input/Questions.csv', encoding='latin-1')
tags = pd.read_csv('../input/Tags.csv', encoding='latin-1')


# In[ ]:


answers = answers.dropna()
questions = questions.dropna()
#tags.dropna()


# In[ ]:


answers.OwnerUserId = answers.OwnerUserId.astype(int)
questions.OwnerUserId = questions.OwnerUserId.astype(int)


# In[ ]:


questions_sample = questions[questions['Score'] >= 75]
tags_sample = tags[tags['Id'].isin(questions_sample['Id'])]


# In[ ]:


result = pd.merge(questions_sample, answers, how = 'inner', left_on = 'Id', right_on = 'ParentId')


# In[ ]:


G = nx.from_pandas_dataframe(result, 'OwnerUserId_x', 'OwnerUserId_y')


# In[ ]:


self_loops = G.selfloop_edges()
G.remove_edges_from(self_loops)
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)


# In[ ]:


print(nx.info(G))


# In[ ]:


nx.average_clustering(G)


# In[ ]:


nx.density(G)


# In[ ]:


centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
tr_clstr = nx.clustering(G)
sq_clstr = nx.square_clustering(G)
eccentricity = nx.eccentricity(G)


# In[ ]:


node = []
deg_cent = []
bet_cent = []
tr_cl = []
sq_cl = []
ecc = []
for v in G.nodes():
    node.append(v)
    deg_cent.append(centrality[v])
    bet_cent.append(betweenness_centrality[v])
    tr_cl.append(tr_clstr[v])
    sq_cl.append(sq_clstr[v])
    ecc.append(eccentricity[v])

node_attr = zip(node, deg_cent, bet_cent, tr_cl, sq_cl, ecc)


# In[ ]:


edges = G.edges()
edgesJson = json.dumps([{'source': source, 'target': target} for source, target in edges], default = str, indent=2, sort_keys=True)  # called a 'list comprehension'


# In[ ]:


nodesJson = json.dumps([{'id': node, 'degree_cent': centrl, 'betweenness_cent': btwn, 'tr_clstr':tr_cl, 'sq_clstr': sq_cl, 'eccentricity': ecc} for node, centrl, btwn, tr_cl, sq_cl, ecc in node_attr], indent=4)


# In[ ]:


tagsJson = json.dumps([{'id': id, 'tag': tag} for id, tag in zip(tags_sample['Id'], tags_sample['Tag'])], default = str, indent=4, sort_keys=True)


# In[ ]:


controlsJson = '[ { "control": "clear", "abbrev": "CLR", "index": 0 }, { "control": "gravity_up", "abbrev": "H G", "index": 1 }, { "control": "gravity_down", "abbrev": "L G", "index": 2 }, { "control": "degree_cent", "abbrev": "DC", "index": 3 }, { "control": "betweenness_cent", "abbrev": "BC", "index": 4 }, {"control":"tr_clstr", "abbrev": "TC", "index" : 5},{ "control": "sq_clstr", "abbrev": "SQC", "index": 6 }, { "control": "eccentricity", "abbrev": "ECC", "index": 7 } ] '


# In[ ]:


Javascript("""
           window.nodes={};
           """.format(nodesJson))


# In[ ]:


Javascript("""
           window.edges={};
           """.format(edgesJson))


# In[ ]:


Javascript("""
           window.tags={};
           """.format(tagsJson))


# In[ ]:


Javascript("""
           window.controls={};
           """.format(controlsJson))


# In[ ]:


get_ipython().run_cell_magic('javascript', '', "require.config({\n    paths: {\n       d3: '//cdnjs.cloudflare.com/ajax/libs/d3/4.8.0/d3.min'\n    }\n});")


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'require([\'d3\'], function(d3){\n    // YOUR CUSTOM D3 CODE GOES HERE:\n    try{ \n        $("#chart1").remove();\n        //create canvas\n        element.append("<div id=\'chart1\'></div>");\n        $("#chart1").width("850px");\n        $("#chart1").height("600px");  \n        var margin = {top: 40, right: 10, bottom: 10, left: 10};\n        var width = 850 - margin.left - margin.right;\n        var height = 600 - margin.top - margin.bottom;\n        var forceCenterOffset = {x: 50, y: 50}\n        var svg = d3.select("#chart1").append("svg")\n            .style("position", "relative")\n            .attr("width", width + "px")\n            .attr("height", (height) + "px")\n            .append("g")\n            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");\n        var simulation = d3.forceSimulation()\n            .force("link", d3.forceLink().id(function(d) { return d.id; }).strength(0.5))\n            .force("charge", d3.forceManyBody().strength(-5))\n            .force("center", d3.forceCenter(width / 2 - forceCenterOffset.x, height / 2- forceCenterOffset.y));\n        var colorScale = d3.scaleLinear().range(["#6b24a5", "#ffffff"]);\n        var strokeWidth = 1.0, cSize = 4;\n        // CONTROL BOXES:\n        var controlBoxes = svg.append("g")\n           .attr("class", "control-boxes")\n           .attr("transform","translate("+margin.left+","+margin.top+")");;\n        // LEGEND & LABELS:\n        svg.append("text")\n            .style("font","14px sans-serif")\n            .attr("transform", "translate(175,0)")\n            .text("Click and hold to highlight connections.  Boxes on the right adjust graph settings.")\n            .style("fill","#000000");\n        var legendSize = {width: 20, height: 200};\n        var colorLegendYScale = d3.scaleLinear().range([0, legendSize.height]);\n        var div = d3.select("#chart1").append("div")\n            .attr("class", "graph-tooltip")\n            .style("opacity", 0)\n            .style("z-index", 1);\n        var colorLegend = svg\n            .append("g")\n            .attr("class", "legend")\n            .attr("transform","translate("+margin.left+","+margin.top+")");\n        var linearGradient = colorLegend.append("defs")\n            .append("linearGradient")\n            .attr("id", "linear-gradient");\n        linearGradient\n          .attr("x1", "0%")\n          .attr("y1", "0%")\n          .attr("x2", "0%")\n          .attr("y2", "100%");\n        linearGradient.selectAll("stop")\n          .data(colorScale.range())\n          .enter()\n          .append("stop")\n          .attr("offset", function(d,i) { return i/(colorScale.range().length-1); })\n          .attr("stop-color", function(d) { return d; })\n          .attr(\'stop-opacity\', 1);\n        colorLegend\n          .append("rect")\n          .attr("width", legendSize.width)\n          .attr("height", legendSize.height)\n          .attr("transform", "translate(0,0)")\n          .style("fill", "url(#linear-gradient)")\n          .style("stroke", "black");\n\n        // DRAW THE SIMULATION:\n        function drawSimulation(nodes, edges){\n            var nodeAttrSelection = "degree_cent";\n            setColorScale(nodes, nodeAttrSelection);\n            var link = svg.append("g")\n              .attr("class", "links")\n              .selectAll("line")\n              .data(edges)\n              .enter()\n              .append("line")\n              .attr("class", "edge")\n              .style("stroke-width", 1.5)\n              .style("stroke", "#bbb");\n            var node = svg.append("g")\n              .attr("class", "nodes")\n              .selectAll("circle")\n              .data(nodes)\n              .enter()\n              .append("circle")\n              .attr("class", "node")\n              .attr("r", cSize)\n              .style("stroke", "black")\n              .style("stroke-width", strokeWidth)\n              .style("fill", function(d){return colorScale(nodeAttrAccessor(d, nodeAttrSelection)); })\n              .call(d3.drag()\n                  .on("start", dragstarted)\n                  .on("drag", dragged)\n                  .on("end", dragended));\n            nodeTooltip(node, nodeAttrSelection);\n            simulation\n              .nodes(nodes)\n              .on("tick", ticked);\n            simulation.force("link")\n              .links(edges);\n            function ticked() {\n            link\n                .attr("x1", function(d) { return d.source.x; })\n                .attr("y1", function(d) { return d.source.y; })\n                .attr("x2", function(d) { return d.target.x; })\n                .attr("y2", function(d) { return d.target.y; });\n            node\n                .attr("cx", function(d) { return d.x; })\n                .attr("cy", function(d) { return d.y; });\n            }\n        };\n\n        // DRAG EVENTS:\n        function dragstarted(d) {\n          if (!d3.event.active) simulation.alphaTarget(0.3).restart();\n          d.fx = d.x;\n          d.fy = d.y;\n          hideOtherNodes(d);\n        }\n        function dragged(d) {\n          d.fx = d3.event.x;\n          d.fy = d3.event.y;\n        }\n        function dragended(d) {\n          if (!d3.event.active) simulation.alphaTarget(0);\n          d.fx = null;\n          d.fy = null;\n          showOtherNodes(d);\n        }\n\n        // CAST THE DATA TYPE TO NUMBER:\n        function castNodeData(nodeData){\n          nodeData.forEach(function(d) {\n            d.degree_cent = +d.degree_cent;\n            d.betweenness_cent = +d.betweenness_cent;\n            d.tr_clstr = +d.tr_clstr;\n            d.sq_clstr = +d.sq_clstr;\n            d.eccentricity = +d.eccentricity;\n          })\n        }\n\n        // HIDE/SHOW NODES ON DRAG:\n        function hideOtherNodes(d){\n          var g_nodes = svg.selectAll(".node");\n          var g_edges = svg.selectAll(".edge");\n          var shownNodes = [];\n          g_edges.filter(function (x) {\n              if (d.id != x.target.id && d.id != x.source.id )\n              {\n                return true;\n              } else {\n                shownNodes.push(x.target.id);    // push ids for nodes connected to dragged node\n                shownNodes.push(x.source.id);\n                return false;\n              }\n            })\n            .style("stroke", "#bbb")\n            .style("stroke-opacity", 0.1)\n            .style();      // fade out everything not connected to dragged node\n          g_edges.filter(function(x){ return d.id === x.target.id || d.id === x.source.id; })\n            .style("stroke", "#000000");\n          g_nodes.filter(function (x) { return (shownNodes.indexOf(x.id) === -1); })\n            .style("fill-opacity", 0.1)\n            .style("stroke-opacity", 0.1)\n            .style("stroke", "#000000")\n            .style("stroke-width", strokeWidth);\n          g_nodes.filter(function (x) { return (shownNodes.indexOf(x.id) != -1); })\n            .style("stroke", "#3039e8")\n            .transition()\n            .duration(200)\n            .style("stroke-width", 2*strokeWidth)\n            .style("r", 2*cSize);\n        };\n        function showOtherNodes(d){\n          var g_nodes = svg.selectAll(".node");\n          var g_edges = svg.selectAll(".edge");\n          g_nodes\n            .style("fill-opacity", 1)\n            .style("stroke-opacity", 1)\n            .transition()\n            .delay(200)\n            .style("r", cSize);\n          g_edges\n          .style("stroke-opacity", 0.6);\n        };\n\n        // VALUE ACCESSOR:\n        function nodeAttrAccessor(d, valueType) {\n          if (valueType === "degree_cent") {\n            return d.degree_cent;\n          } else if ( valueType === "betweenness_cent") {\n            return d.betweenness_cent;\n          } else if ( valueType === "tr_clstr") {\n            return d.tr_clstr;\n          } else if (valueType === "sq_clstr") {\n            return d.sq_clstr;\n          } else if (valueType === "eccentricity") {\n            return d.eccentricity;\n          }\n        }\n\n        // CONTROLS:\n        function drawControls(controls, nodes){\n          var transitionDuration = 75;\n          var controlBoxSize = 30;\n          var controlBoxScaleUp = 1.33;\n          var controlXOffset = 200;\n          var g_box = controlBoxes\n            .selectAll("g")\n            .data(controls)\n            .enter()\n            .append("g")\n            .attr("transform", function (d,i){\n              return "translate("+(width - controlXOffset)+","+(i*(controlBoxSize+ 5))+")"\n            })\n            .attr("class", "controls");\n          g_box\n            .append("rect")\n            .attr("class", "control")\n            .attr("width", controlBoxSize)\n            .attr("height", controlBoxSize)\n            .style("stroke",  function(d){\n              if (d.control === "clear") {\n                return "#3039e8";\n              } else {\n                return "black";\n              }\n            })\n            .style("fill", function(d){\n              if (d.control === "clear") {\n                return "#ffffff";\n              } else if (d.control === "gravity_up" || d.control === "gravity_down") {\n                return "#b8b9bc"\n              } else {\n                return "#b592d2"\n              }\n             });\n          g_box\n            .append("text")\n            .attr("x", 0.08*controlBoxSize)\n            .attr("y", 0.6*controlBoxSize)\n            .text(function(d){ return d.abbrev ;})\n            .style("pointer-events","none")\n            ;\n          g_box\n            .selectAll("rect")\n            .on("click", function(d){\n              if (d.control === "clear") {\n                resetNodeBorder();\n              } else if (d.control === "gravity_up") {\n                changeGravity("up");\n              } else if (d.control === "gravity_down") {\n                changeGravity("down");\n              } else {\n                setNodeAttribute(d.control, nodes);\n              }\n            })\n            .on("mouseover", function(d, i){\n              d3.select(this)\n                .transition()\n                .duration(transitionDuration)\n                .attr("width", controlBoxSize*controlBoxScaleUp)\n                .attr("height", controlBoxSize*controlBoxScaleUp)\n                .style("stroke-width", 2);\n                var index = d.index, additionalOffset = (controlBoxScaleUp-1)*controlBoxSize;\n              g_box\n                .transition()\n                .duration(transitionDuration)\n                .attr("transform", function (d,i){\n                  if ( i > index) {\n                    return "translate("+(width - controlXOffset)+","+(i*(controlBoxSize+5)+additionalOffset)+")"\n                  } else {\n                    return "translate("+(width - controlXOffset)+","+(i*(controlBoxSize+5))+")"\n                  }\n                })\n                controlTooltip(g_box, index);\n            })\n            .on("mouseout", function(d){\n              d3.select(this)\n                .transition()\n                .duration(transitionDuration)\n                .attr("width", controlBoxSize)\n                .attr("height", controlBoxSize)\n                .style("stroke-width", 1);\n              g_box\n                .transition()\n                .duration(transitionDuration)\n                .attr("transform", function (d,i){\n                    return "translate("+(width - controlXOffset)+","+(i*(controlBoxSize+ 5))+")"\n                })\n            });\n        };\n        // CONTROL FUNCTIONS:\n        function resetNodeBorder(){\n          var g_nodes = svg.selectAll(".node");\n          var g_edges = svg.selectAll(".edge");\n          g_nodes\n            .style("stroke", "#000000")\n            .style("stroke-width", strokeWidth);\n          g_edges\n            .style("stroke", "#bbb")\n        };\n\n        function changeGravity(direction){\n          if (direction ==="down") {\n            simulation.force("charge", d3.forceManyBody().strength(-25));\n            simulation.alphaTarget(0.3).restart();\n            setTimeout(function() { simulation.alphaTarget(0); }, 2500);\n          } else if (direction === "up") {\n            simulation.force("charge", d3.forceManyBody().strength(-5));\n            simulation.alphaTarget(0.3).restart();\n            setTimeout(function() { simulation.alphaTarget(0); }, 2500);\n          } else {\n            simulation\n              .force("charge", d3.forceManyBody().strength(-5));\n          }\n        };\n\n        function setNodeAttribute(attributeType, nodes){\n          setColorScale(nodes, attributeType);\n          var node = svg.selectAll("circle.node")\n            .style("fill", function(d){return colorScale(nodeAttrAccessor(d, attributeType)); });\n          nodeTooltip(node, attributeType);\n        };\n\n        function setColorScale(nodes, attributeType){\n          var nodeAttrMax = d3.max(nodes, function(d){ return nodeAttrAccessor(d, attributeType);});\n          //var nodeAttrMin = 0;\n          var nodeAttrMin = d3.min(nodes, function(d){ return nodeAttrAccessor(d, attributeType);});  \n          var nodeAttrScaleAdj = (nodeAttrMax - nodeAttrMin)\n          var nodeAttrExtent = [(nodeAttrMax-nodeAttrScaleAdj*0.25), (nodeAttrMin)];\n          colorScale.domain(nodeAttrExtent);\n          setLegendScale(nodes, attributeType, nodeAttrExtent)\n        };\n\n        function nodeTooltip(node, nodeAttrSelection){\n          node\n            .on("mouseover", function(d) {\n              div.transition()\n                  .duration(200)\n                  .style("opacity", .9);\n              div.html("id:"+d.id+ "<br/>" +nodeAttrSelection+": " + nodeAttrAccessor(d, nodeAttrSelection).toFixed(5))\n                 .style("left", (d3.event.pageX) + "px")\n                 .style("top", (d3.event.pageY) + "px");\n              console.log("x: "+d3.event.pageX+"; y: "+d3.event.pageY);\n              })\n          .on("mouseout", function(d) {\n              div.transition()\n                  .duration(500)\n                  .style("opacity", 0);\n          });\n        };\n\n        function controlTooltip(cBox, index){\n          var tooltipHTML = "";\n          switch(index) {\n            case 0:\n              tooltipHTML = "Clear Selection";\n              break;\n            case 1:\n              tooltipHTML = "High Gravity";\n              break;\n            case 2:\n              tooltipHTML = "Low Gravity";\n              break;\n            case 3:\n              tooltipHTML = "Show Degree Centrality";\n              break;\n            case 4:\n              tooltipHTML = "Show Betweenness Centrality";\n              break;\n            case 5:\n              tooltipHTML = "Show Triangle Clustering";\n              break;\n            case 6:\n              tooltipHTML = "Show Square Clustering";\n              break;\n            case 7:\n              tooltipHTML = "Show Eccentricity";\n              break;\n          }\n          cBox\n            .on("mouseover", function(d) {\n              div.transition()\n                  .duration(200)\n                  .style("opacity", .9);\n              div.html(tooltipHTML)\n                 .style("left", (d3.event.pageX) + "px")\n                 .style("top", (d3.event.pageY) + "px");\n              console.log("x: "+d3.event.pageX+"; y: "+d3.event.pageY);\n              })\n          .on("mouseout", function(d) {\n              div.transition()\n                  .duration(500)\n                  .style("opacity", 0);\n          });\n        };\n\n        // COLOR LEGEND FUNCTIONS:\n        function setLegendScale(data, nodeAttrSelection, colorDomain){\n          colorLegendYScale.domain(colorDomain);\n          var colorLegendYAxis = d3.axisRight(colorLegendYScale);\n          colorLegend\n                .selectAll(".y.axis")\n                .remove(); \n          colorLegend\n                .selectAll(".label")\n                .remove();\n          colorLegend\n                .append("g")\n                .attr("class","y axis")\n                .attr("transform", "translate(25,0)");\n          colorLegend.selectAll(".y.axis")\n                .call(colorLegendYAxis)\n                .append("text")\n                .attr("class", "tick")\n                .attr("transform", "rotate(-90)")\n                .attr("y", 6)\n                .attr("dy", ".71em")\n                .style("text-anchor", "end");\n          colorLegend\n                .append("g")\n                .attr("class", "label")\n                .attr("transform", "translate(-5,200)")\n                .append("text")\n                .style("font","14px sans-serif")\n                .attr("transform", "rotate(-90)")\n                .text(nodeAttrSelection)\n                .style("fill","#000000");\n        };\n        \n        function setCSS(){\n            d3.select("#chart1")\n              .style("font", "10px sans-serif");\n\n            controlBoxes.selectAll("text")\n              .style("pointer-events", "none");\n\n            d3.select("div.graph-tooltip")\n              .style("position", "relative")\n              .style("text-align","center")\n              .style("width","180px")\n              .style("height","28px")\n              .style("padding","2px")\n              .style("font","12px sans-serif")\n              .style("background","lightsteelblue")\n              .style("border","0px")\n              .style("border-radius","4px")\n              .style("pointer-events","none");\n            \n        };\n\n        /* ************************************************************** */\n        // MAIN:\n        /* ************************************************************** */\n        // GET THE NETWORK DATA AND CALL DRAW FUNCTION\n        var nodesData = window.nodes;\n        var edgesData = window.edges;\n        var controlsData = window.controls;\n        \n        setCSS();\n        castNodeData(nodesData);\n        drawSimulation(nodesData, edgesData);\n        drawControls(controlsData, nodesData);\n    } catch(err) {\n        console.log("Viz Error: ");\n        console.log(err);\n    }\n});')

