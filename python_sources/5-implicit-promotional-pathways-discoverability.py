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
#     <li><a href="https://www.kaggle.com/shivamb/1-description-structuring-engine-cola">Part 1: Job Bulletin Structuring Engine - City of Los Angeles </a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola">Part 2: Encourage Diversity and Remove Unconsious Bias from Job Bulletins - A Deep Analysis</a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola">Part 3: Impact of Content, Tone, and Language : CTL Analysis for CoLA</a>  </li> 
#     <li><a href="https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola">Part 4: Increasing the Discoverability of Promotional Pathways (Explicit)  </a></li>
#     <li><a href="https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola">Part 5: Implicit Promotional Pathways Discoverability</a></li></ul>
# <div align='center'><font size="5" color="#ff3fb5">Part 5: Implicit - Promotional Pathways Discoverability</font></div>
# <div align='center'>Other Parts: <a href='https://www.kaggle.com/shivamb/1-description-structuring-engine-cola'>Part 1</a> | <a href='https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola'>Part 2</a> | <a href='https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola'>Part 3</a> | <a href='https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola'>Part 4</a> | <a href='https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability'>Part 5</a> </div>
# <p style='text-align:justify'>In the last kernel, I explored the method to identify and visualize promotional pathways which are mentioned explicitly in the job requirements. In this kernel, I have made an attempt to identify the promotional pathways using the contextual anlaysis of job requirements (implict). I have shared the methodology below: </p>       
# 
# **Methodology:**   
# > 1. From the structured data, obtain the complete requirement text and perform basic text cleaning.   
# > 2. Represent every requirement text as a vector in which the context and semantics are preserved. I have used Pre-Trained Word Embeddings using [fasttext](https://fasttext.cc/) for this purpose.  
# > For every job class requirement, representing it as a vector is very helpful as the word embedding vectors can be used to identify other job classes which shares similar requirements. In cases when a job class is not mentioned explicitly in the requirment, this method can be used to identify implict links.   
# > 3. Compute a contextual similarity matrix which gives similarity scores of one class with the others.  
# > 4. Use the similarity matrix to identify possible candidates. Filter them using a dictionary of seniority levels and flexible ngram matching to ensure that parent job class is actually linked to a child job class. 
# 
# The overview of the methodology is shown in the following process flow diagram. 
# 
# <br>
# ![](https://i.imgur.com/WDzTuSQ.png)
# <br>
# 
# There are two parts in this method:   
# 
# A: Pre-Processing Stage : Compute requirement context vectors, and context similarity matrix   
# B: Identification Stage : Finding the implicit links using similarity scores, dictionary, and ngram matching. 
# 
# Following are the contents of the kernel: 
# 
# ## <font color="#ff3fb5">Contents:</font>  
# 
# <a href='#1'>1. Load Pre-Trained Word Embeddings</a>   
# <a href='#2'>2. Load and Clean the Requirements Text Data</a>   
# <a href='#3'>3. Convert Requirements to Requirements Context Vectors</a>   
# <a href='#4'>4. Compute Contextual Similarity Matrix</a>   
# <a href='#5'>5. Identify and Filter the Implicit Links</a>   
# <a href='#6'>6. Write the Visualization Functions</a>   
# <a href='#7'>7. Examples</a>   
# 
# <div id="1"></div>
# ## <font color="#ff3fb5">1. Load Pre-Trained Word Embeddings</font>  
# 
# A popular idea in modern machine learning is to represent words by vectors (also called word embeddings). These vectors capture hidden information about a language, like word analogies or semantic. Let's load the 2M word embedding vectors in a python object from fasttext dataset. 

# In[ ]:


from tqdm import tqdm 
import numpy as np 

embeddings_index = {}
EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'
f = open(EMBEDDING_FILE, encoding="utf8")
for line in tqdm(f):
    values = line.split()
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[values[0]] = coefs
f.close()


# <div id="2"></div>
# ## <font color="#ff3fb5">2. Load and Clean the Requirements Text  </font>  
# 
# Next, we load and perform text clenaing on the requirement texts of job bulletins. We can either load directly from files, or use the structured file generated in Kernel 1. 

# In[ ]:


from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords 
import pandas as pd
import string, os
import json

ignorewords = ["year", "experience", "full-time", "part-time", "part", "time", "full", "university", "college", "degree", "major"]
stopwords = stopwords.words('english')
numbs  = "0123456789"

""" function to cleanup the text """
def _cleanup(text):
    text = text.lower()
    text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
    text = " ".join([c for c in text.split() if c not in stopwords])
    words = []
    for wrd in text.split():
        if len(wrd) <= 2: 
            continue
        if wrd in ignorewords:
            continue
        words.append(wrd)
    text = " ".join(words)    
    return text

""" function to clean the filename and obtain the job role title"""
def _clean(fname):
    for num in numbs: 
        fname = fname.split(num)[0].strip()
    return fname.title()

results = []
base_path = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"
for fname in os.listdir(base_path):
    if fname == "POLICE COMMANDER 2251 092917.txt":
        continue

    txt = open(base_path + fname).read()
    lines = txt.split("\n")
    start = 0
    rel_lines = []
    for i, l in enumerate(lines):
        if 'requirement' in l.lower():
            start = i
            break
    for i, l in enumerate(lines[start+1:]):
        if "substituted" in l.lower():
            break
        if l.isupper():
            break
        rel_lines.append(l)
    req1 = " ".join(rel_lines)
    req = _cleanup(req1)
    d = {'cleaned' : req, 'original' : req1, 'title' : _clean(fname)}
    results.append(d)
    
data = pd.DataFrame(results)[['title','original','cleaned']]
data.head()


# <div id="3"></div>
# ## <font color="#ff3fb5">3. Convert Requirements to Requirement Context Vector </font>  
# 
# Now, we will write a function to convert the requirement text into a vector form. This vector maintains the content, context, and semantics of the original text. The idea of generating a document vector from word vectors is to perform simple aggregation.

# In[ ]:


""" function to generate document vector by aggregating word embeddings """
def generate_doc_vectors(s):
    words = str(s).lower().split() 
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        if w in embeddings_index:
            M.append(embeddings_index[w])
    v = np.array(M).sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

req_vectors = []
for i,r in data.iterrows():
    req_vectors.append(generate_doc_vectors(r['cleaned']))


# <div id="4"></div>
# ## <font color="#ff3fb5">4. Compute Contextual Similarity between the Requirements </font>  
# 
# Next, we perform the pairwise similarity between the vectors. For this purpose, I have used [linear_kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html) from the python's sklearn package. The output of this snippet is a matrix which stores the similarity between any two vectors in the entire dataset. Additionally, I have also iterated for every job class, and identified top 20 most contextually similar classes. 

# In[ ]:


_interactions = linear_kernel(req_vectors, req_vectors)

_edges = {}
for idx, row in data.iterrows():
    similar_indices = _interactions[idx].argsort()[:-100:-1]
    similar_items = [(_interactions[idx][i], data['title'][i]) for i in similar_indices]
    _edges[row['title']] = similar_items[:20]


# <div id="5"></div>
# ## <font color="#ff3fb5">5. Identify the Implicit Connections between requirements </font>  
# 
# Now, we need to ensure that the parent class is one of the higher classes (director, principle, chief, senior etc) and child class is one of the lower class. In this function, formatting of links is also performed which makes it easier for visualization purposes.

# In[ ]:


""" function to identify the implicit links """
def get_treedata(k, threshold, limit):
    k = k.title()

    txt = """<h3><font color="#aa42f4">Requirement Texts: </font></h3>"""
    txt += "<h3><font color='#196ed6'>" + k + "</font></h3>"
    req = data[data['title'] == k]['original'].iloc(0)[0]
    if len(req) > 350:
        req = req[:350] + " ..."
    txt += "<p><b>Requirements: </b>" + req + "</p>"

    treedata = {"name" : k, "children" : [], "color" : '#97f4e3', "size":25, "exp" : ""}
    edges = _edges[k]
    edges = [_ for _ in edges if _[1] != k]
    edges = [_ for _ in edges if _[0] >= threshold]
    ignore = ['principal', "chief", "director", "supervisor"]
    counter = 0
    for i, edge in enumerate(edges):
        if any(upper in edge[1].lower() for upper in ignore):
            continue
        d = {"name" : edge[1], "children" : [], "color" : "red", "size":15, "exp" : edge[0]}
        treedata['children'].append(d)
        counter += 1
        if counter == limit:
            break
        txt += "<h3><font color='#f93b5e'>" + edge[1] + "(Context Similarity: "+str(round(edge[0], 2))+")</font></h3>"
        req1 = data[data['title'] == edge[1]]['original'].iloc(0)[0]
        if len(req1) > 350:
            req1 = req1[:350] + " ..."
        txt += "<b>Requirements: </b>" + req1 + ""
    return treedata, txt


# > - The methodology works for most of the classes but at the same time it might gives wrong results for some classes. This is possibly because of similarity of words which are not important to a role (example - experience, applicant etc.)  
# 
# <div id="6"></div>
# ## <font color="#ff3fb5">6. Visualizing the Implict Links </font>  
# 
# In the next cell, I have shared the code to visualize the links. 

# In[ ]:


from IPython.core.display import display, HTML, Javascript
import IPython.display

""" function to generate required javascript and HTML for the visualization """
def _get_js(treedata, idd):
    html = """<style>  
        .node circle {
          fill: #fff;
          stroke: steelblue;
          stroke-width: 3px;
        }
        .node text { font: 12px sans-serif; }
        .node--internal text {
          text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;
        }
        .link {
          fill: none;
          stroke: #ccc;
          stroke-width: 2px;
        }
    </style>
    <svg height='340' id='"""+idd+"""' width="760"></svg>"""

    js="""require.config({
        paths: {
            d3: "https://d3js.org/d3.v4.min"
        }
    });
    require(["d3"], function(d3) {
        var treeData ="""+json.dumps(treedata)+""";

        // set the dimensions and margins of the diagram
        var margin = {top: 40, right: 90, bottom: 50, left: 90},
            width = 660 - margin.left - margin.right,
            height = 290 - margin.top - margin.bottom;

        // declares a tree layout and assigns the size
        var treemap = d3.tree()
            .size([width, height]);

        //  assigns the data to a hierarchy using parent-child relationships
        var nodes = d3.hierarchy(treeData);

        // maps the node data to the tree layout
        nodes = treemap(nodes);

        // append the svg obgect to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        var svg = d3.select('#"""+idd+"""').append("svg")
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
               return "M" + d.x + "," + d.y
                 + "C" + d.x + "," + (d.y + d.parent.y) / 2
                 + " " + d.parent.x + "," +  (d.y + d.parent.y) / 2
                 + " " + d.parent.x + "," + d.parent.y;
               });

        // adds each node as a group
        var node = g.selectAll(".node")
            .data(nodes.descendants())
          .enter().append("g")
            .attr("class", function(d) { 
              return "node" + 
                (d.children ? " node--internal" : " node--leaf"); })
            .attr("transform", function(d) { 
              return "translate(" + d.x + "," + d.y + ")"; });

        // adds the circle to the node
        node.append("image")
        .attr("xlink:href", function(d) { return "https://image.flaticon.com/icons/png/512/306/306473.png" })
        .attr("x", function(d) { return -15;})
        .attr("y", function(d) { return -15;})
        .attr("height", 30)
        .attr("width", 30);

        // adds the text to the node
        node.append("text")
          .attr("dy", ".35em")
          .attr("y", function(d) { return d.children ? -20 : 20; })
          .style("text-anchor", "middle")
          .text(function(d) { return d.data.name; })
          .attr("transform", "rotate(-10)" );        
    });"""
    
    return html, js

def _implicit(title, idd, threshold=0.88, limit = 4):
    treedata, txt = get_treedata(title, threshold, limit)
    h, js = _get_js(treedata, idd)
    h = display(HTML(h))
    j = IPython.display.Javascript(js)
    IPython.display.display_javascript(j)
    display(HTML(txt))


# <div id="7"></div>
# ## <font color="#ff3fb5">7. Some examples </font>  
# 
# Now, let's look at some examples of job classes and the implicit links between them. 
# 
# ### 7.1 Senior Administrative Clerk

# In[ ]:


_implicit("Senior Administrative Clerk", idd="a8", threshold = 0.75, limit = 6)


# > - In this example of "Senior Administrative Clerk" we can observe the position requires 1 year of clerical experience. We see that in related jobs which have contextually similar requirements also asks for similar experience. Secretary - 1 year of clerical experience, Accounting clerk - 2 years of clerical work, and so on. 
# > - This means that the employees who are serving as these positions and have spent enough experience, are eligible to be promoted as Senior Administrative Clerk role. 
# 
# ### 7.2 Chief Benefits Analyst

# In[ ]:


_implicit("Chief Benefits Analyst", idd="a1")


# > - The requirement texts of these job roles asks for similar work experience, hence employees who are retirement plana manager roles can also become analysts after spending considerable amount of experience required. 
# 
# ### 7.3 EMS Nurse Practitioner Supervisor

# In[ ]:


_implicit("Ems Nurse Practitioner Supervisor", idd="a2")


# ### 7.4 Helicopter Mechanic Supervisor

# In[ ]:


_implicit("HELICOPTER MECHANIC SUPERVISOR", idd="a3")


# In this example as well, we see that specific job class names are not mentioned in the requirement texts however the work to be done or the responsibilities are almost similar. Hence their exist an implict link between the two classes. 
# 
# ### 7.5 General Automotive Supervisor

# In[ ]:


_implicit("General Automotive Supervisor", idd="a4")


# ### 7.6 Steam Plant Maintenance Supervisor

# In[ ]:


_implicit("Steam Plant Maintenance Supervisor", idd="a5")


# ### 7.7 Wastewater Treatment Electrician Supervisor

# In[ ]:


_implicit("Wastewater Treatment Electrician Supervisor", idd="a6")


# ## <font color="#ff3fb5">End Notes</font>
# 
# This was the last kernel of my submission. Hope you have gone through all the other parts. If there are any questions about my entire solution, please feel free to share them in the comments. If you liked it upvote. I really enjoyed working on this competition. Other Links: <a href='https://www.kaggle.com/shivamb/1-description-structuring-engine-cola'>Part 1</a> | <a href='https://www.kaggle.com/shivamb/2-encourage-diversity-reduce-bias-cola'>Part 2</a> | <a href='https://www.kaggle.com/shivamb/3-impact-of-ctl-content-tone-language-cola'>Part 3</a> | <a href='https://www.kaggle.com/shivamb/4-promotional-pathway-discoverability-cola'>Part 4 </a> | <a href='https://www.kaggle.com/shivamb/5-implicit-promotional-pathways-discoverability/'> Part 5 </a>   
# 
# Thanks. 
# 
# -- Shivam
