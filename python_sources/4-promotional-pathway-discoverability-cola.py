#!/usr/bin/env python
# coding: utf-8

# <div align='center'><font size="6" color="#ff3fb5">Data Science For Good : CoLA</font></div>
# <div align='center'><font size="4" color="#ff3fb5">A Complete Pipeline for Structuring, Analysis and Recommendation</font></div>
# <div align='center'><font size="3" color="#ff3fb5">Improve Hiring Process and Decisions</font></div>
# <hr>
# 
# <p style='text-align:justify'><b>Key Objectives:</b> Keeping these challenges in mind, an ideal solution for the City of Los Angeles has following key objectives: Develop an nlp framework to accurately structurize the job descriptions. Develop an analysis framework to identify the implict bias in the text and encourage diversity. Develop a system which can clearly identify the promotion pathways within the organization.</p>
# 
# <b>My Submission:</b> Following are parts of Kernels Submissions in order:<br>
# 
# <ul>
#     <li><a href="https://www.kaggle.com/shivamb/1-description-structuring-engine-cola" target="_blank">Part 1: Job Bulletin Structuring Engine - City of Los Angeles </a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola" target="_blank">Part 2: Encourage Diversity and Remove Unconsious Bias from Job Bulletins - A Deep Analysis</a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola" target="_blank">Part 3: Impact of Content, Tone, and Language : CTL Analysis for CoLA</a>  </li> 
#     <li><a href="https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola" target="_blank">Part 4: Increasing the Discoverability of Promotional Pathways (Explicit) </a></li>
#     <li><a href="https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability/" target="_blank">Part 5: Implicit Promotional Pathways Discoverability</a></li></ul>
# 
# <div align='center'><font size="5" color="#ff3fb5">Part 4: Increasing the Discoverability of Promotional Pathways </font></div>
# <div align='center'>Other Parts: <a href='https://www.kaggle.com/shivamb/1-description-structuring-engine-cola'>Part 1</a> | <a href='https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola'>Part 2</a> | <a href='https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola'>Part 3</a> | <a href='https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola'>Part 4</a> | <a href='https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability/'>Part 5</a></div>
# <p style='text-align:justify'>The aim of this kernel is to provide an easy solution to identify the explict promotion pathways in City of Los Angeles. I developed a reusable python module in which the program identifies which job roles are required to fill another particular job class. Using this class, one can identify:</p>       
#      
# - What are the possible pathways of Job Class which are required to fill a particular Job Class?     
# - What are the possible Job Classes an Employee can be promoted to? 
# 
# ### <font color="#ff3fb5">Table of Contents</font>   
# 
# <a href="#1">1. Possible Pathways to fill a Job Class </a>          
# <a href="#2">2. Possible Promotion Pathways for a Job Class </a>        
# 
# 
# <div id="1"></div>
# ## <font color="#ff3fb5">1. Possible Pathways to fill a Class</font> 
# 
# Sometimes, it is very hard to find qualified candidates to fill a particular job class. Infact, City of Los Angeles has mentioned that there are atleast 17 job classes int which it is very challenging to find the right qualified candidates. These include Accountant, Accounting Clerk, Applications Programmer, Assistant Street Lighting Electrician, Building Mechanical, Inspector, Detention Officer, Electrical Mechanic, Equipment Mechanic, Field Engineering Aide, Housing Inspector, Housing Investigator, Librarian, Security Officer, Senior Administrative Clerk, Senior Custodian, Senior Equipment Mechanic, Tree Surgeon. 
# 
# The following framework analyzes the structured data, specifically the requirements of a particular job and identifies which other job classes are required to fill the particular job class. In the end, City of LA can use this piece of code to generate all possible promotion pathways using a single line of code. 

# In[ ]:


from IPython.core.display import display, HTML, Javascript
import IPython.display
import json, os
import random 

""" define global variables, maps, and lookup dictionaries to be used in the program """
colors = ['red', 'blue','green', 'yellow', 'pink', 'gray', 'orange', 'purple', 'brown', 'silver','mistygreen','chocolate', 'peru', 'indigo', 'deeppink', 'linen', 'lavender', 'snow', 'indianred', 'lawngreen', 'papayawhip', 'mediumturquoise', 'plum', 'tan', 'magenta', 'wheat', 'aqua', 'cadetblue', 'forestgreen', 'honeydew', 'ivory', 'orchid']
colors = ["#80f2e6", "#84f984", "#9e86ef", "#f9b083", "#ed82ab", "#ff5661", "#82cef2", "#ef91f2", "#fcf344", "#b59f98", "#727171", "#a87000", "#a2a800", "#7ba800", "#00a80b", "#1d78c6", "#535e77", "#dbd9f9", "#58286d", "#b100ff", "#ff2dd1", "#dbb1d"]
n_map  = {"one" : "1", "two" : "2", "three" : "3" , "four" : "4", 'five' : "5", "six" : "6", "seven" : "7"}
levels = ['X', 'IX', 'VIII', 'VII', 'VI', 'V', 'IV', 'III', 'II', "I"]
path   = "../input/cityofla/CityofLA/Job Bulletins/"
flags  = ['as a', 'the level of', ' as ']
numbs  = "0123456789"
files  = os.listdir(path)

random.shuffle(colors)

""" function to clean the filename and obtain the job role title"""
def _clean(fname):
    for num in numbs: 
        fname = fname.split(num)[0].strip()
    return fname.lower()

""" function to clean the position-level from a job role title """
def gclean(x):
    for level in levels:
        x = x.replace(" "+level+" ","").strip()
    return x

""" get the value of year from the experience related requirement """
def getyear(txt):
    year = ""
    if "university" in txt:
        txt = txt.split("university")[1]
    txt = txt.split("year")[0]
    if "." in txt:
        txt = txt.split(".")[1]
    elif "and " in txt:
        txt = txt.split("and ")[1]
    if txt.strip().lower() in n_map:
        year = n_map[txt.strip().lower()]
    return year
    
""" function to search and extract the job roles mentioned in the requirement line related to experience """
def _get_roles_required(fname):
    
    ## extract the requirement portion
    txt = open(path + fname).read()
    lines = txt.split("\n")
    for i, x in enumerate(lines):
        if "requirement" in x.lower():
            start = i
            break
    for i, x in enumerate(lines[start+1:]):
        if x.isupper():
            end = i
            break
    
    ## check if any role is mentioned
    results = {}
    years = []
    for i, x in enumerate(lines[start+1:start+end]):
        if not any(_ in x for _ in flags):
            continue
        for r in roles:
            for level in levels:
                if r.title() + " " + level in x:
                    relt = x.split(r.title()+" "+level)[0]
                    results[r.title() + " " + level] = getyear(relt)
                elif r.title() in x:
                    relt = x.split(r.title())[0]
                    results[r.title()] = getyear(relt)
    
    ## remove redundant roles 
    removals = []
    keys = list(results.keys())
    for l1 in keys:
        for l2 in keys:
            if l1 == l2:
                continue
            if l1 in l2:
                removals.append(l1)
    for rem in list(set(removals)):
        del results[rem]
    return results

## create a global mapping of all the roles and their required job roles
roles = [_clean(f) for f in files]
roles = [x for x in roles if x]

doc = {}
for f in files:
    if _clean(f) == "":
        continue
    try:
        res = _get_roles_required(f)
        doc[_clean(f).title()] = res
    except Exception as E:
        pass
    
""" function to return node size based on experience """  
def map_size(num):
    if num == "":
        return 10
    num = int(num)
    doc = {1:10, 2:15, 3:23, 4:32, 5:39, 6:28}    
    if num in doc:
        return doc[num]
    else:
        return 40
    
""" function to generate the relevant data for the tree like visualization """
def get_tree_data(key, ddoc):
    visited = {}
    cnt = 0
    v = ddoc[key]
    ## level 0
    
    treedata = {"name" : key, "children" : [], "color" : '#97f4e3', "size":40, "exp" : ""}
    for each in v:
        exp = str(v[each]) + "y"
        size = map_size(v[each])
        
        if key == each:
            continue
        
        ## level 1
        c_each = gclean(each)
        if c_each not in ddoc:
            if c_each not in visited:
                visited[each] = colors[cnt]
                cnt += 1
            d = {"name" : each, "children" : [], "color" : visited[each], "size":size, "exp" : exp}
            treedata['children'].append(d)
        
        if c_each in ddoc:
            if each not in visited:
                visited[each] = colors[cnt]
                cnt += 1
            
            d = {"name" : each, "children" : [], "color" : visited[each], "size":size, "exp" : exp}
            for each1 in ddoc[c_each]:
                if each1 not in visited:
                    visited[each1] = colors[cnt]
                    cnt += 1
                
                exp = str(ddoc[c_each][each1]) + "y"
                size = map_size(ddoc[c_each][each1])
                m = {"name" : each1, "children" : [], "color" : visited[each1], "size":size, "exp" : exp}
                
                ## level 2 
                c_k = gclean(each1)
                if c_k in ddoc:
                    for each2 in ddoc[c_k]:
                        if each2 not in visited:
                            visited[each2] = colors[cnt]
                            cnt += 1
                        
                        exp = str(ddoc[c_k][each2]) + "y"
                        size = map_size(ddoc[c_k][each2])
                        p = {'name' : each2, "children" : [], "color" : visited[each2], "size":size, "exp" : exp}
                        m['children'].append(p)
                d['children'].append(m)
            treedata['children'].append(d)
    return treedata

""" function to generate required javascript and HTML for the visualization """
def _get_js(treedata, div_id, rot, small = False):
    rt = ""
    if rot == True:
        rt = """.attr("transform", "rotate(-10)" )"""
    ht = 610
    if small:
        ht = 360
    
    html = """<style>  
        .node circle {
         stroke-width: 4px;
        }
        .node text { font: 11px sans-serif; }
        .node--internal text {
         text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;
        }
        .link {
         fill: none;
         stroke: #ccc;
        }
    </style>
    <svg height='"""+str(ht)+"""' id='"""+div_id+"""' width="760"></svg>"""

    js="""require.config({
        paths: {
            d3: "https://d3js.org/d3.v4.min"
        }
    });
    require(["d3"], function(d3) {
        var treeData = """ +json.dumps(treedata)+ """;

        var margin = {top: 40, right: 30, bottom: 50, left: 90},
            width = 760 - margin.left - margin.right,
            height = '"""+str(ht-50)+"""' - margin.top - margin.bottom;

        var treemap = d3.tree().size([width, height]);
        var nodes = d3.hierarchy(treeData);
        nodes = treemap(nodes);
        var svg = d3.select('#"""+div_id+"""').append("svg").attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom),
            g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");
        var link = g.selectAll(".link").data(nodes.descendants().slice(1)).enter().append("path").attr("class", "link").attr("d", function(d) {
            return "M" + d.x + "," + d.y + "C" + d.x + "," + (d.y + d.parent.y) / 2 + " " + d.parent.x + "," + (d.y + d.parent.y) / 2 + " " + d.parent.x + "," + d.parent.y;
        }).attr("stroke-width", function(d) {
            return (7);
        });
        var node = g.selectAll(".node").data(nodes.descendants()).enter().append("g").attr("class", function(d) {
            return "node" + (d.children ? " node--internal" : " node--leaf");
        }).attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
        node.append("circle").attr("r", function(d){ return d.data.size }).style("fill", function(d) {
            return d.data.color;
        });
        
        node.append("text")
              .attr("text-anchor", "middle")
              .attr("dy", ".35em")
              .text(function(d) { return d.data.exp; });
        node.append("text").attr("dy", ".15em").attr("y", function(d) {
            return d.children ? -20 : 20;
        }).style("text-anchor", "middle").text(function(d) {
            var name = d.data.name;
            return name;
        })"""+rt+""";
        
    });"""
    return html, js

def getjob(key, idd, rot, small = False):
    treedata = get_tree_data(key, doc)
    html, js = _get_js(treedata, idd, rot, small)
    h = display(HTML(html))
    j = IPython.display.Javascript(js)
    IPython.display.display_javascript(j)


# In the last cell, all the code is developed. Now, we need to call one function to view the visual representation of all possible pathways to fill a particular job class. Let's look at some examples. 
# 
# **Legend:** 
# - Color : Every node's color represents one particular job class. ie. Same job Classes have same color  
# - Size : Experience required (in years) before they can be promoted.    
# - Higher the node size, higher experience is required. Number is also written in node center.   
# 
# These graphs are made using d3.js which is executed very simply using python.  
# 
# ### 1. Possible Pathways for "Senior Systems Analyst"

# In[ ]:


getjob("Senior Systems Analyst", "id1", rot=False)


# > - This graph suggest that, Employees who have served 3 years as a Senior Administrative Clerk or four years as a Custoemr Service Representative can be promoted for the role of Systems Aide. 
# > - Having spent 2 years as Systems Aide, they can be promoted for the role of Systems Analyst where they have to spend minimum 2 years before promoting to Senior Systems Analyst Role.   
# 
# ### 2. Possible Pathways for "Chief of Airport Planning"

# In[ ]:


getjob("Chief Of Airport Planning", "id2", rot=False)


# > - In this example, The employees who are Planning Assistant, if they have spent 2 years in the role, then they are eligible to be promoted as City Planning Associate. If they have spent 4 years in the role, they are eligible to be promoted as City Planner. Both these nodes are shown in Green. 
# 
# 
# ### 3. Possible Pathways for "Water Utility Superintendent"

# In[ ]:


getjob("Water Utility Superintendent", "id3", rot=True)


# > - In this example, to become **Water Utility Supervisor**, any employee having the role as Supervisor related to Water Service, Water Treatment, Waterworks Mechanic, Water Utility, or Water Utility Operator. ( LEVEL 1 nodes)     
# > - (Blue Node) Water Utility Worker, can either first become Water Service Worker (Green Node), and then either promoted to Water Service Supervisor or Water Utility Supervisor. Otherwise, they can also be directly promoted to Supervisor position, skipping the Service Worker class.  
# 
# Let's look at another complicated example, but made simpler using visualization. 
# 
# ### 4. Possible Pathways for "Chief Inspector"

# In[ ]:


getjob("Chief Inspector", "id4", rot=True)


# > - This example suggest that there are many pathways to fill a role of Chief Inspector. Assistant Inspector can follow atleast 6 different pathways to get the first promotion and subsequently find more. 
# 
# Let's also look at some of the roles which are difficult to fill 
# 
# ### 5. "Senior Equipment Mechanic" and "Housing Inspector"

# In[ ]:


getjob("Senior Equipment Mechanic", "id11", rot=False, small = True)
getjob("Housing Inspector", "id12", rot=False, small = True)


# ## <font color="#ff3fb5">How to Generate these Plots?</font> 
# 
# These graphs for a job role can be generate using a single line of the code by calling the function: 
# 
# > getjob("JobTitle", "AnyID", rot=True, small=False)
# 
# This function accepts four arguments: 
# 
# > - "First Argument" is the Job title   
# > - "Second Argument" is a ID given to the plot, (any string)   
# > - "rot" : if the title text should be rotated in the graph (True / False)   
# > - "small" : if the height of the plot should be small or large (True / False)
# 
# Let's look at some examples - 

# In[ ]:


getjob("Senior Housing Inspector",      "plot_id2", rot=True, small = True)
getjob("Management Analyst",            "plot_id3", rot=True, small = True)
getjob("Director Of Printing Services", "plot_id4", rot=True, small = False)


# <h1><a href='http://www.shivambansal.com/blog/network/cola/explict_pathways.html'> VIEW PROMOTIONAL PATHWAYS for ALL CLASSES</a></h1>
# 
# 
# <div id="2"></div>
# ## <font color="#ff3fb5">2. Possible Promotion Pathways for a Class</font> 
# 
# We need to develop a system in which given an employee with a particular job class and a given experience, what are the possible pathways they can take to get promoted. In other words, identifying all the ways in which the employees are eligible for promotions. 

# In[ ]:


rdoc = {}
for k,v in doc.items():
    for each in v:
        if each not in rdoc:
            rdoc[each] = []
        if each != k:
            rdoc[each].append(k)
            
""" function to generate the relevant data for the tree like visualization """
def get_tree_data_promotion(key, ddoc):
    visited = {}
    cnt = 0
    v = ddoc[key]
    ## level 0
    
    treedata = {"name" : key, "children" : [], "color" : '#97f4e3', "size":40, "exp" : ""}
    for each in v:
        if key == each:
            continue
        
        ## level 1
        c_each = gclean(each)
        if c_each not in ddoc:
            if c_each not in visited:
                visited[each] = colors[cnt]
                cnt += 1
            d = {"name" : each, "children" : [], "color" : visited[each]}
            treedata['children'].append(d)
        
        if c_each in ddoc:
            if each not in visited:
                visited[each] = colors[cnt]
                cnt += 1
            
            d = {"name" : each, "children" : [], "color" : visited[each]}
            for each1 in ddoc[c_each]:
                if each1 not in visited:
                    visited[each1] = colors[cnt]
                    cnt += 1
                
                m = {"name" : each1, "children" : [], "color" : visited[each1]}
                
                ## level 2 
                c_k = gclean(each1)
                if c_k in ddoc:
                    for each2 in ddoc[c_k]:
                        if each2 not in visited:
                            visited[each2] = colors[cnt]
                            cnt += 1
                        
                        p = {'name' : each2, "children" : [], "color" : visited[each2]}
                        m['children'].append(p)
                d['children'].append(m)
            treedata['children'].append(d)
    return treedata


def _get_js2(treedata, div_id, rot):
    rt = ""
    if rot == True:
        rt = """.attr("transform", "rotate(-20)" )"""
    
    html = """
    <svg height="510" id='"""+div_id+"""' width="860"></svg>"""

    js="""require.config({
        paths: {
            d3: "https://d3js.org/d3.v4.min"
        }
    });
    require(["d3"], function(d3) {
        var treeData = """ +json.dumps(treedata)+ """;
        
    // set the dimensions and margins of the diagram
    var margin = {top: 20, right: 130, bottom: 30, left: 120},
        width = 660 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    // declares a tree layout and assigns the size
    var treemap = d3.tree()
        .size([height, width]);

    //  assigns the data to a hierarchy using parent-child relationships
    var nodes = d3.hierarchy(treeData, function(d) {
        return d.children;
      });

    // maps the node data to the tree layout
    nodes = treemap(nodes);

    var svg = d3.select('#"""+div_id+"""').append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom),
        g = svg.append("g")
          .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

    // adds the links between the nodes
    var link = g.selectAll(".link")
        .data( nodes.descendants().slice(1))
      .enter().append("path")
        .attr("class", "link")
        .attr("d", function(d) {
           return "M" + d.y + "," + d.x
             + "C" + (d.y + d.parent.y) / 2 + "," + d.x
             + " " + (d.y + d.parent.y) / 2 + "," + d.parent.x
             + " " + d.parent.y + "," + d.parent.x;
           }).attr("stroke-width", function(d) {
            return (7);
        });

    // adds each node as a group
    var node = g.selectAll(".node")
        .data(nodes.descendants())
      .enter().append("g")
        .attr("class", function(d) { 
          return "node" + 
            (d.children ? " node--internal" : " node--leaf"); })
        .attr("transform", function(d) { 
          return "translate(" + d.y + "," + d.x + ")"; });

    // adds the circle to the node
    node.append("circle")
      .attr("r", function(d) { return 12; })
      .style("stroke", function(d) { return 2; })
      .style("fill", function(d) { return d.data.color; });

    // adds the text to the node
    node.append("text")
      .attr("dy", ".35em")
      .attr("x", function(d) { return d.children ? 
        (d.data.value + 4) * -1 : d.data.value + 4 })
      .style("text-anchor", function(d) { 
        return d.children ? "end" : "start"; })
      .text(function(d) { return d.data.name; })"""+rt+""";;

     });"""

    return html, js

def getjob_junior(key, idd, rot):
    treedata = get_tree_data_promotion(key, rdoc)
    html, js = _get_js2(treedata, idd, rot)
    h = display(HTML(html))
    j = IPython.display.Javascript(js)
    IPython.display.display_javascript(j)


# ## 2.1 How can "City Planner" be promoted ?

# In[ ]:


getjob_junior("City Planner".title(), "id7", rot=True)


# > - This graph is read from left to right. An employee who is currently working as a City Planner, can be first promoted to Chief of Airport Planning or Senior City Planner. Once they have served the required experience as a Senior City Planner, they can further be promoted to next levels which are - Principle City Planner or Associate Zoning Administrator
# 
# ## 2.2 In what ways a "Police Officer" can be promoted? 

# In[ ]:


getjob_junior("Police Officer".title(), "id9", rot=True)


# ## 2.3 "Electrical Mechanic" Promotion Pathways

# In[ ]:


getjob_junior("Electrical Mechanic".title(), "id10", rot=True)


# ## 2.4 "Secretary" - All possible promotion paths

# In[ ]:


getjob_junior("Secretary", "id5", rot=True)


# > - A Secretary can be promoted to Executive Administrative Assistant or Commission Executive Assistant. Once the person has served as the Executive Administrative Assistant, they can be further promoted to Management Analyst which is a very good role. This is because as a Management Analyst and having served enough eperience the person can be promoted to more than 10 different roles as shown in the graph. 
# 
# ## <font color="#ff3fb5">How to Generate these Plots?</font>
# 
# These graphs for a job role can be generate using a single line of the code by calling the function: 
# 
# > getjob_junior("JobTitle", "AnyID", rot=True)
# 
# This function accepts three arguments: 
# 
# > - "First Argument" is the Job title   
# > - "Second Argument" is a ID given to the plot, (any string)   
# > - "rot" : if the title text should be rotated in the graph (True / False)   
# 
# and to generate graphs to explore the other view, with years and node size, one can use the first function. 
# 
# > getjob("JobTitle", "AnyID", rot=True, small=False)
# 
# This function accepts only an extra argument: small = True / False to control the height of the chart.  
# 
# 
# ### Next Kernels: 
# For next parts of my submission (analysis and recommendations), please visit next kernels of my Submission: 
# <ul>
#     <li><a href="https://www.kaggle.com/shivamb/1-description-structuring-engine-cola" target="_blank">Part 1: Job Bulletin Structuring Engine - City of Los Angeles </a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola" target="_blank">Part 2: Encourage Diversity and Remove Unconsious Bias from Job Bulletins - A Deep Analysis</a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola" target="_blank">Part 3: Impact of Content, Tone, and Language : CTL Analysis for CoLA</a>  </li> 
#     <li><a href="https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola" target="_blank">Part 4: Increasing the Discoverability of Promotional Pathways (Explicit) </a></li>
#     <li><a href="https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability/" target="_blank">Part 5: Implicit Promotional Pathways Discoverability</a></li></ul>
# 
# Next <a href="https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability/" target="_blank">Kernel</a> - Implict promotional pathways visualization and technique.
