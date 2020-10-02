#!/usr/bin/env python
# coding: utf-8

# # Attaching Images to Notebook
# 
# Created nice code, and wanting to make your notebook public? Congratulations on you getting to that point!
# Now before sharing, juice up your notebook just a bit with some images. Add just that final bit of polish onto your creation. You can do this easily. One of the methods below will teach you how just a line of code!
# 
# **Ways of attaching Images**:
# 1. [HTML / Markdown](#section-one)
# 2. [Python Image Library](#section-two)
# 3. [Javascript](#section-three)
# 
# I recommend using either the first two, they are far by the easiest to implement.

# <a id="section-one"></a>
# # 1. HTML / Markdown
# My favourite method is this one. Just incoporate this HTML code in a *Markdown* cell:
# 
# `<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52"/>`
# 
# Alternatively: this markdown code has a similar effect. Thanks to @leifuer for pointing this one out. 
# 
# `![Image](https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png)`
# 
# Below you will see the result. This is the picture header for the learn together beginners' competition in Kaggle. Always do acknowledge where you are referencing your image for courtesy.

# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52"/>
# 

# So you can reference even pictures that are stored on your website. "src" refers to source.
# To use your own image, just follow this format:
# 
# `<img src="https://yoursite.com/yourpic.png/>`
# 
# or 
# 
# `![Image](https://yoururl.com)`
# 
# 
# Note: The code only works in a markdown cell and not a code cell.

# <a id="section-two"></a>
# # 2. Python Image Library
# 
# Now this one involves actual use of some Python code. It is slightly more involved but nothing that you can't implement within 5 minutes.
# I have uploaded an image into this notebook via the "+ Add Data" button on the top right.
# You will see this button when you copy and edit this notebook for yourself.

# In[ ]:


from IPython.display import Image
Image("../input/bricklayer-mason-plasterer-worker-cartoon-100266268.jpg")


# Congratulations! With these two lines of codes, you can also add images to your notebook.
# Pretty simple and easy to implement!
# 
# Photo by vectorolie. Published on 11 June 2014 | Stock photo - Image ID: 100266268 | freedigitalphotos.net

# <a id="section-three"></a>
# # 3. Javascript
# 
# This is where it gets more advanced. I would only recommend this if you have experience in coding with Javascript. The advantage of using this method is that you are able to add your own customizations.

# In[ ]:


from IPython.core.display import display, HTML, Javascript
import IPython.display

html_string = """
<g id="my_image_goes_here"></g>
"""

js_string = """
require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

require(["d3"], function(d3) {
  
  d3.select("#my_image_goes_here")
      .append("img")
      .attr("src", "http://www.freedigitalphotos.net/images/previews/bricklayer-mason-plasterer-worker-cartoon-100266268.jpg")
      .attr("width", "250px")
      .style("border", "10px solid black");

});
"""

h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)


# # Conclusion
# 
# Now that you have learnt all these techniques, get yourself out there and start beautifying your public notebooks! 
# Look forwards to seeing what you have created and do share with me.
# 
# Should you be interested in some other quick tips, check out:
# https://www.kaggle.com/dcstang/create-table-of-contents-in-a-notebook
# 
# As always, feel free to fork this notebook. If this helped you out, please drop an upvote for this notebook just under this cell! 
