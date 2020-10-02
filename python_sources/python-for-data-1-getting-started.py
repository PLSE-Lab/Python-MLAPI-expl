#!/usr/bin/env python
# coding: utf-8

# # Python for Data 1: Getting Started
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# This notebook is the first in an ongoing series of 30 lessons aimed at providing an introduction to the Python programming language for data analysis and predictive modeling. This Kaggle guide is an updated version of a [30-part Python for Data Analysis](https://hamelg.blogspot.com/2015/12/python-for-data-analysis-index.html) guide I created on my blog with the same general structure. This guide does not assume prior experience with programming or data science and focuses on the use of Python as a tool for data analysis.

# # About Python

# Python is an [open source](https://en.wikipedia.org/wiki/Open_source) programming language that is popular for general software development and scripting, web development and data science. Since Python is a general purpose language, the base language does not include many of the data structures and functions necessary for data analysis. Python does, however, have an extensive selection of addon packages that give it great flexiblity and a robust library of data-oriented addons make it one of the most popular languages for data science. The majority of the programming kernels you'll find on Kaggle are written in Python.
# 
# If you're wondering whether you should start your data science journey by learning Python or R, there's no correct answer. Both languages are used extensively in data science and it a good idea to learn the basics of both. Python is the most popular langauge for data science and most Kaggle users recommend learning it first. R is great for data exploration, statistics and plotting, as many of the functions you need are built into the language. If you are interested in learning more about the pros and cons of learning data science with R vs Python, you can read [this blog post](https://hamelg.blogspot.com/2016/08/python-vs-r-for-data-science.html) I wrote on the subject. If you'd prefer to start by learning R for data analysis, you can check out my [Intro to R](https://www.kaggle.com/hamelg/intro-to-r-index).

# # Python Setup

# This guide uses the Kaggle kernel environment to deliver content, so you do not need to set up a local Python environment to follow along. Instead, you can simply read each lesson in your browser window and if you'd like to run and interact with the code, you can fork the kernel for a given lesson to create a personal copy of it that you can alter and run as you see fit. In reworking the guide for the Kaggle kernel environment I hope that it becomes more accessible and interactive while simultaneously avoiding problems with incompatibilities across different versions of Python.
# 
# If you would like to set up a local Python environment on your computer, you will need to download Python. It is easiest to download the Anaconda Python distribution from Continuum Analytics. Anaconda bundles Python with dozens of popular data analysis libraries and it comes with a nice integrated development environment (a fancy code editor) called Spyder. See [Part 1](http://hamelg.blogspot.com/2015/10/python-for-data-analysis-part-1-setup.html) of my original guide for a brief tutorial on setting up Python locally. This guide does not require that you have a local Python environment and assumes you'll be following along on Kaggle's kernels instead.

# # Using Kaggle Notebooks

# The remainder of this lesson will consist of a short tutorial on how to use the Kaggle notebooks to follow along with this guide. Although you can read along without interacting with the code, I suggest that you fork notebooks for lessons of interest so that you can experiment and run code yourself. For each lesson, I may add some programming exercises to give you a chance to practice using concepts from the lesson; doing these exercises will require forking the notebook.
# 
# To fork a Kaggle notebook, you will need a Kaggle account and then to click on the blue "Copy and Edit" button at the top right of the notebook page. This will create a copy of the notebook on your account that you can interact with. Try it now to follow along with the rest of this notebook!

# The programming notebooks Kaggle uses, known as Jupyter notebooks, provide a handy way of structuring text and code in a single document that can be rendered as HTML--a web page--so that it can be viewed in your web browser. Programming notebooks consist of two types of cells: Markdown and Code. Markdown cells contain plain text that can be given additional structure using a text formatting language called [Markdown](https://en.wikipedia.org/wiki/Markdown). Code cells consist of code that you can run interactively while you are editing the notebook. Clicking on any part of the notebook while you are in edit mode after forking it will highlight the cell containing the text or code. Click on this line of text to select this Markdown cell! 
# 
# Now that you have selected this Markdown cell, you should notice some formatting tools in the upper left corner of the cell that allow you to perform some common formatting tasks like adding web links and making lists. On the right side, you will see a tab that says Markdown in blue, followed by Code in gray. The word Markdown in blue indicates that this is currently a Markdown text cell. You can convert Markdown cells to code cells and back by clicking on the appropriate word on the tab. Try selecting cell below and then convert it from a Markdown cell to a code cell:

# #### Turn this cell into a code cell!
# 5 + 10

# Code cells differ from Markdown cells in that you can run code cells, which causes whatever code they contain to be executed and then whatever output the code produces to appear below the cell. After selecting a code cell, hold down the control key and then press enter to run its contents (you can also click the blue "play" button to the left of the cell while it is selected to run it.). Try running the cell above that you converted from Markdown to code!
# 
# The code cell above should have run the arithmetic operation 5 + 10 and produced an output value of 15. Congratulations, you've run your first Python code! 
# 
# If you change the contents of a code cell and run it again, the output will change according to the new code. Try changing the code in the cell above from "5 + 10" to "5 * 10" and run it again. The output should now be 50, since you changed the operation from addition to multiplication.

# # Notebook Shortcuts

# The Jupyter programming environment comes with a variety of keyboard shortcuts that can help you do common operations quicker. First of all, it should be noted that the Jupyter notebook has two distinct modes: Edit mode and Command mode. When you click on a Markdown or code cell like this one (click on it now!), the notebook will be put into edit mode allowing you to directly edit the contents of the cell. While in edit mode, you have access to keyboard shortcuts that deal with editing the contents of the selected cell. Going into command mode lets you use a variety of higher level commands that let you do things like create new cells, delete cells and convert cells from one type to another. 
# 
# While in edit mode, press the escape key to enter command mode. Try it now! 
# 
# You should notice the flashing text edit cursor disappear when you enter command mode. Now that you are in command mode, press the "P" key to bring up a list of notebook commands and then search for "show keyboard shortcuts" and click on it. You'll notice two lists: one for keyboard commands that work in edit mode and one for commands that work while you are in command mode. Don't be overwhelmed by the number of keyboard shortcuts: it is not really important that you learn any of them for this guide beyond using control + enter to run code, but I wanted you to be aware that you can use various shortcuts to help navigate the notebooks. 
# 
# It should be noted that control + enter to run code cells works in both edit mode and command mode. Some other useful shortcuts while in command mode include: "A" to create a new cell above the current cell, "B" to create a new cell below the current cell, "M" to convert the current cell to Markdown, "Y" to convert the current cell to code and "DD" (press "D" twice) to delete the current cell.

# # Wrap Up

# This lesson is just the tip of the iceberg. Although we spent most of our time discussing the programming environment we will use throughout this guide, we are now well prepared to jump in and start learning Python. In the following lessons we will continue our journey by learning the very basics of Python and programming in general to give you the foundation necessary to move onto using Python as a tool for data analysis and predictive modeling. We will start slow, but during the course of this guide you will learn all the tools you need to start exploring data, generating plots and making predictive models with Python. We will use the Titanic disaster data set as our primary motivating example, but the tools you learn will be transferable to any data project you're looking to tackle. If you have some prior experience with Python or another programming language you may wish to [skip ahead](https://www.kaggle.com/hamelg/python-for-data-analysis-index) to a specific part of this guide that interests you.

# # Next Lesson: [Python for Data 2: Python Arithmetic](https://www.kaggle.com/hamelg/python-for-data-2-python-arithmetic)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
