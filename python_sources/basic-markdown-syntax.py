#!/usr/bin/env python
# coding: utf-8

# <h1><center><font color = 'blue' face = 'times'>MARKDOWN BASICS</font></center></h1>
# <h5>By - Vishnu Prakash Singh</h5>
# <h6>01st Jan,2019</h6>

# ### Notes:
# * Convert the code cell into markdown cell using the drop down box in task bar.
# * Same can be done using shortcut 'm' in command mode or esc + 'm' in edit mode.
# * In each cell of the tutorial, codes is above and corresponding output is below in the same cell.

# ---
# ### 1.1) Writing in Italics
# ```python
# *I am writing this line in italic using markdown syntax 1.*
# _I am writing this line in italic  using markdown syntax 2._
# <i>I am writing this line in italic using HTML syntax.</i>
# # Add <br> at the end of sentence to break it if more than one line is being copy pasted at a time.
# ```
# *I am writing this line in italic using markdown syntax 1.*<br>
# _I am writing this line in italic  using markdown syntax 2._<br>
# <i>I am writing this line in italic using HTML syntax.</i>

# ------
# ### 1.2) Writing in Bold
# ```python
# **I am writing this line in bold using markdown syntax 1.**
# __I am writing this line in bold using markdown syntax 2.__
# <b>I am writing this line in bold  using HTML syntax.</b>
# ```
# **I am writing this line in bold using markdown syntax 1.**<br>
# __I am writing this line in bold using markdown syntax 2.__<br>
# <b>I am writing this line in bold  using HTML syntax.</b>

# ---
# ### 1.3) Writing Code
# ```python
# To read a csv file, use `pd.read_csv`.
# To wrte a csv file, use <code>pd.to_csv</code>.
# ```
# To read a csv file, use `pd.read_csv`.<br>
# To wrte a csv file, use <code>pd.to_csv</code>.

# ---
# ### 1.4) Writing in Blockquotes
# ```python
# > I am learning markdown basics.
# >> Markdown is a good way to present.
# ```
# > I am learning markdown basics.
# >> Markdown is a good way to present.

# ------
# ### 1.5) Underlining Text in Markdown
# ```python
# <u>I am underlining this line</u>
# ```
# <u>I am underlining this line</u>

# ---
# ### 1.6) Strike Through Texts
# ```python
# ~~I am striking through this line.~~
# ```
# ~~I am striking through this line.~~

# ---
# ### 1.7) Citing in Markdown
# ```python
# <cite>I am citing this line</cite>
# ```
# <cite>I am citing this line</cite>

# ---
# ### 1.8) Breaking a line in Markdown
# ```python
# This is a very long line,<br> i should break it into two.
# ```
# This is a very long line,<br> i should break it into two.

# ---
# ### 1.9) Breaking a line in Markdown
# ```python
# This is a very long line,<hr> i should break it into two by drawing a line in between.
# ```
# This is a very long line,<hr> i should break it into two by drawing a line in between.

# ---
# ### 2.1) Ordered Lists in Markdown
# ```python
# __I like following things__
# 1. Movies
#     1. More Movies
#     2. TV Series
# 2. Coffee
# # instead of *, + , - can also be used.
# ```
# __I like following things__
# 1. Movies
#     1. More Movies
#     2. TV Series
# 2. Coffee

# ---
# ### 2.2) Ordered Lists in Markdown using HTML syntax
# ```python
# 
# __I like following things__
# <ol>
# <li> Movies</li>
# <ol>
# <li> More Movies</li>
# <li>  TV Series</li>
# </ol>
# <li> Coffee</li>
# </ol>
# ```
# __I like following things__
# <ol>
# <li> Movies</li>
# <ol>
# <li> More Movies</li>
# <li>  TV Series</li>
# </ol>
# <li> Coffee</li>
# </ol>

# ---
# ### 2.2) Unordered Lists in Markdown
# ```python
# __I like following things__
# * Movies
#     * More Movies
#     * TV Series
# * Coffee
# # instead of *, + , - can also be used.
# ```
# ---
# __I like following things__
# * Movies
#     * More Movies
#     * TV Series
# * Coffee

# ---
# ### 2.4) Unordered Lists in Markdown using HTML syntax
# ```python
# __I like following things__
# <ul>
# <li>Movies
# <ul>
# <li>More Movies</li>
# <li>TV Series</li>
# </ul>
# <li>Coffee</li>
# </ul>
# ```
# ---
# __I like following things__
# <ul>
# <li>Movies
# <ul>
# <li>More Movies</li>
# <li>TV Series</li>
# </ul>
# <li>Coffee</li>
# </ul>

# ---
# ### 2.5) Lists with check box in Markdown
# ```python
# - [ ] Do's
#     - [ ] Eat
#     - [ ] Sleep
# - [x] Dont's
#     - [x] Kill
# ```
# - [ ] Do's
#     - [ ] Eat
#     - [ ] Sleep
# - [x] Dont's
#     - [x] Kill

# ---
# ### 3.1) Headings in Markdown
# ```python
# # This is Heading 1.
# <h1>This is Heading 1.</h1>
# ```
# # This is Heading 1.

# ---
# ### 3.2) Headings in Markdown
# ```python
# ## This is Heading 2.
# <h2>This is Heading 2.</h2>
# ```
# ## This is Heading 2.

# ---
# ### 3.3) Headings in Markdown
# ```python
# ### This is Heading 3.
# <h3>This is Heading 3.</3>
# ```
# ### This is Heading 3.

# ---
# ### 3.4) Headings in Markdown
# ```python
# #### This is Heading 4.
# <h4>This is Heading 4.</4>
# ```
# #### This is Heading 4.

# ---
# ### 3.5) Headings in Markdown
# ```python
# ##### This is Heading 5.
# <h5>This is Heading 5.</5>
# ```
# ##### This is Heading 5.

# ---
# ### 3.6) Headings in Markdown
# ```python
# ###### This is Heading 6.
# <h6>This is Heading 6.</6>
# ```
# ###### This is Heading 6.

# ---
# ### 4.1) Aligning Texts in Markdown
# ```python
# <center>This Text is put in center.</center>
# ```
# <center>This Text is put in center.</center>

# ---
# ### 4.2) Aligning Headings in Markdown
# ```python
# <h3><center>This Heading is put in center.</center></h3>
# ```
# <h3><center>This Heading is put in center.</center></h3>

# ---
# ### 5.1) Code Highlighting in Markdown
# 
# ````
# ```python
# print("Hello World!!!")
# ```
# ````
# 
# ```python
# print("Hello World!!!")
# ```

# ---
# ### 5.2) Code Highlighting in Markdown
# 
# ````
# ```java
# print("Hello World!!!")
# ```
# ````
# 
# ```java
# print("Hello World!!!")
# ```

# ---
# ### 6.1) Inserting Table in Markdown
# ```python
# |  Col1| Col2 |
# |------|------|
# |This  | is   |
# |   a  | table|
# ```
# 
# |  Col1| Col2 |
# |------|------|
# |This  | is   |
# |   a  | table|

# ----
# ### 6.2) Inserting Formulae
# ```python
# $e^{i\pi} + 1 = 0$
# ```
# $e^{i\pi} + 1 = 0$

# ---
# ### 6.3) Inserting Links
# ```python
# [github](https://github.com/, 'github.com')
# # Hover over 'github' to see 'github.com'
# ```
# [github](https://github.com/, 'github.com')

# ---
# ### 6.4) Inserting Local Images
# ```python
# ![Alt sin_cos_curve](my_figure.png "Sine Vs Cos Curve")
# # Drag and drop the image in a markdown cell to add an image in python notebook.
# # In order to view image in pynb/HTML, image must be present at the location.
# # Hover over the image for the title text
# ```
# ![Alt sin_cos_curve](attachment:my_figure.png "Sine Vs Cos Curve")

# ---
# ### 6.5) Inserting Local Images using code cell
# In this case, no need to keep image at specific location.
from IPython.display import Image;
Image('my_figure.png',width=400,height = 200)
# ---
# ### 6.6) Inserting Online Images
# ```python
# ![Python Logo](https://jelastic.com/blog/wp-content/uploads/2019/02/python-logo.png "Logo of Python")
# ```
# ![Python Logo](https://jelastic.com/blog/wp-content/uploads/2019/02/python-logo.png "Logo of Python")

# ---
# ### 6.7) Inserting horizontal line
# ```python
# ---
# ***
# ---
# <hr size = 2 height = 3 width = 300>
# ```
# ---
# 
# <hr size = 5 height = 5 width = 300>
# 

# ### 6.8) Inserting shapes
# ```python
# I will display &#9658;
# I will display &#9632;
# #for more shapes, refer - https://www.w3schools.com/charsets/ref_utf_geometric.asp
# ```
# 
# I will display &#9658;<br>
# I will display &#9632;
# 

# ---
# ### 7.1) Changing Color of text
# ```python
# <font color='green'>The color of this text is green.</font>
# ```
# <font color='green'>The color of this text is green.</font>

# ---
# ### 7.2) Changing Size of text
# ```python
# <font size= 5>The size of this text is 5.</font>
# ```
# <font size= 5>The size of this text is 5.</font>

# ---
# ### 7.3) Changing font of text
# ```python
# <font face="arial">The font of this text is Arial.</font>
# # For more fonts - https://websitesetup.org/web-safe-fonts-html-css/
# ```
# <font face="arial">The font of this text is Arial.</font>

# ---
# ### 7.4) Changing color, size, font of text
# ```python
# <font size="4" face="Verdana" color="blue">
# This text is in Verdana, size 4, and in blue color.
# </font>
# ```
# <font size="4" face="Verdana" color="blue">
# This text is in Verdana, size 4, and in blue color.
# </font>

# ---
# ### 7.5) Increasing Size of text
# ```python
# <font size="+2">I am increasing the size of the text by 2.</font>
# ```
# <font size="+2">I am increasing the size of the text by 2.</font>

# ---
# ### 7.6) Decreasing Size of text
# ```python
# <font size="-1">I am decreasing the size of the text by 1.</font>
# ```
# <font size="-1">I am decreasing the size of the text by 1.</font>

# ---
# ### 8.1) Escaping Special Characters
# ```python
# \* I want to wrap this line in star symbol. \*
# \` I want to wrap this line in backticks. \`
# 
# # Use backslash(\) before the special character to tell Jupyter notebook to ignore Markdown Syntax.
# # Backslash can escape these special characters  `, *, _,{}, [], (), #, +, -, ., !, \
# ```
# \* I want to wrap this line in star symbol. \*<br>
# \` I want to wrap this line in backticks. \`

# ---
# ### 8.2) Other way of escaping Backticks
# 
# ```python
# ```` `I want to wrap this line in backticks.` ```` ```
# ```python
# # Use ```` to the left and right of the text to escape backticks in markdown 
# ```
# 
# ````
# ` I want to wrap this line in backticks.`
# ````

# # <center>***THE END***</center>
# ---
