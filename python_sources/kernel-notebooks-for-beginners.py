#!/usr/bin/env python
# coding: utf-8

# # Kernel Notebooks for Beginners
# 
# Welcome to this tutorial on using notebook kernels for your data analysis and data science work. These kernels are based off [Jupyter notebooks](https://jupyter.org/), the industry standard tool for working with data and developing models. Notebooks provide a way to include text, math expressions, images, videos, and code all in one shareable document. Notebooks are also exceptional pedagogical tools for teaching data science and machine learning.
# 
# Kernels come with a coding environment where the most used data science packages and libraries are pre-installed for you. You can also easily add datasets to use in the kernel. This makes is easy to start up a kernel notebook and get right to working with your data, iterating on models, and winning competitions. Kernels also provide GPUs for training deep neural networks.
# 
# However, if you're a beginner or used to working in IDEs, notebooks can be difficult to use at first. So, this kernel is intended to teach you the basics of using notebooks. Kernel notebooks also provide more functionality than Jupyter notebooks, so I'll also show you the fun things we've built.
# 
# ## Fork this Kernel
# 
# One of my favorite things about kernels is that you can "fork" someone else's kernel and modify their code. Forking here means that you get a copy of your own that you can edit and run. This tutorial will be interactive so you'll need to fork this kernel and continue on in your own forked version. So, please click on the blue Fork button in the top right corner.

# ## Cells
# 
# Notebooks are contructed of two kinds of cells, **Markdown cells** and **code cells**. Markdown cells are where the text and such goes, what you're reading right now is a Markdown cell. Code cells run code.
# 
# Notebooks are in general language agnostic. They were first built to run Julia, Python, and R, which is where the name Jupyter comes from. Since then the community has built extensions (also called kernels) that run dozens of other languages. On your local machine, you must choose which language to use when you create a notebook. These kernel notebooks run Python 3 or R which you can select in the Settings panel in the right sidebar.
# 
# To execute a cell and move to the next cell, with the cell selected, press Shift + Enter. You can also run a cell and *not* move using Ctrl + Enter. Select the code cell below and press Shift + Enter.

# In[1]:


x = 2
y = 3
x + y


# You'll notice the result of the code is printed out below the cell. In notebooks, the output of the final line is printed without the need to use the `print` function. This is often useful for viewing the results of some computation.

# You can switch cells between code and Markdown modes by pressing your Escape key then either the `y` or `m` keys. `y` changes a Markdown cell to a code cell, `m` changes a code cell to a Markdown cell. Try it on this cell! Hit Escape then switch between cell modes with `y` and `m`. It's important to get used to using key shortcuts to be productive with notebooks.

# Pressing Escape moves you from **editing mode** to **command mode**. In editing mode, you are editing a cell while in command mode you can use key commands such as `y` and `m`. To move to editing mode from command mode, you can click on a cell or press Enter.
# 
# Other useful key commands:
# * `a` creates a new cell above the currently selected cell
# * `b` creates a new cell below the currently selected cell
# * `c` copies a cell
# * `x` cuts a cell
# * `v` pastes a cell
# * `dd`, that's `d` two times in row, deletes a cell
# 
# Try adding, deleting, and copy-cut-pasting cells below. In Kernels, you can also create cells by clicking on the +Code and +Markdown buttons below. Remember to go back to command mode using the Escape key.

# In[ ]:


# Try out key commands here.


# ## Code cells
# 
# In code cells you write your code! Any variables, functions, modules, etc you create in one cell are available in all other cells. This includes cells that occur earlier in the notebook. This can lead to errors and confusion when your code expects the cells to be run out of order. In general, you want to do your best to make sure all the code runs currectly from the beginning of the notebook to the end.
# 
# **Tip!** You can turn on line numbers by entering command mode on a code cell and pressing `l`.
# 
# Run the cell below, then in an empty code cell multiply `x` and `y`.

# In[ ]:


x = 5
y = 8


# In[ ]:


# multiply x and y in this cell


# ## Markdown cells
# 
# Markdown cells (like this one) are where you can add text, images, and other media to the notebook. This is often useful for providing documentation and descriptions for the code. Writing these descriptions is extremely helpful for others reading your kernels and also for you when you come back to something after time away.
# 
# **Tip:** Feel free to edit these Markdown cells to get used to how this works. To render a Markdown cell, press Shift + Enter. To edit a rendered cell, click on it.
# 
# Markdown is a standard markup language that makes it easy to add formatting to your text, without have to type out HTML tags. You can read the [syntax documentation here](https://daringfireball.net/projects/markdown/syntax). 
# 
# For example, to make text *emphasized* place a single asterisk around the text, like \*emphasized\*. To make text **strong/bold**, use two asterisks like \*\*strong/bold\*\*.
# 
# 
# ### Quotes
# For block quoted text, add a right caret > before the text. It'll be formatted something like this, one of my favorite quotes from Kurt Vonnegut
# > And I urge you to please notice when you are happy, and exclaim or murmur or think at some point, 'If this isn't nice, I don't know what is.'
# 
# 
# ### Code formatting
# You can format text as code using backticks so \`import numpy\` is rendered as `import numpy`. To format a block of code use three backticks at the beginning and end of the block.
# ```python
# def lower_split(string):
#     return string.lower().split()
# ```
# 
# Language specific syntax highlighting is provided by defining the language after the first three backticks. Starting a code block with \`\`\`python will highlight Python code appropriately. The code block above is written in Markdown as
# <pre>
# ```python  
# def lower_split(string):
#     return string.lower().split()  
# ```
# </pre>
# 
# 

# ### Math
# Notebooks can also display math expressions using LaTeX notation by placing dollar signs \$ around the expression. LaTeX is the standard format for defining math expressions and is used widely in the scientific community. Notebooks specifically use a library called Mathjax to render the LaTeX code as math expressions.
# 
# An inline expression such as $e^{i\pi} + 1 = 0$ is written as `$e^{i\pi} + 1 = 0$`. You can do blocked math expressions (sometimes called "display mode") using two dollars signs \$\$ at the beginning and end of the math expression. So this Markdown
# 
# <pre>
# $$
# e^{ix} = \cos(x) + i \sin(x)
# $$
# </pre>
# 
# looks like 
# 
# $$
# e^{ix} = \cos(x) + i \sin(x)
# $$
# 
# Math expressions are extremely useful when describing algorithms, data processing, models, etc. [Here is a good resource](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference) for learning how to write LaTeX expressions in notebooks.

# ### Links
# 
# You can create links to websites using with brackest and parentheses like so \[some text that will be the link\](https://www.kaggle.com), which renders to [some text that will be the link](https://www.kaggle.com).
# 
# Images are similar, but you put a \! at the beginning like 
# ```
# ![alt text](http://url_to_image.com)
# ```

# ### Arbitrary HTML
# 
# Markdown also accepts HTML tags so you can use something like `<img src=http://url_to_image.com width=400>` if you need to.
# 
# ### Kernel Markdown editor
# 
# As you've probably noticed, kernels provide a convenient toolbar for formatting your Markdown at the top of the cell. You won't see this in a default Jupyter notebook on your own machine though, so it's good to get used to typing Markdown.

# ## Kaggle Kernel specific things
# 
# Kaggle kernels have some great features for working with data and developing analysis or modeling code. Firstly, when you are editing a kernel, you can add data by clicking on the `+ Add Dataset` button in the top right. You can connect to a dataset already on Kaggle, or upload your own. These datasets are placed in the `input` folder so you can access them there. You should see the Workspace file structure in the sidebar on the right.
# 
# If you are working with deep learning models, it's often useful to accelerate training with a GPU. Kaggle kernels provide free access to a GPU which you can turn on by clicking Settings in the sidebar and then clicking on the GPU button.
# 
# When you hover over a cell, you can see a little menu of buttons on the top right. You can use these to move cells up and down, delete the cell, collapse the cell, or more options.
# 
# ### Saving your work and publishing
# 
# As you work in a kernel your edits are automatically saved as a draft. When you feel like the kernel is in a good place and ready for people to view it, you can press the Commit button to publish the kernel. Kaggle keeps track of versions if you commit multiple times. This way anyone (including yourself) can see changes you've made to the notebook as you develop it. This is similar to version control with git if you've used it before. To make the kernel public so others can see, share, and fork it, set Sharing to Public in the sidebar's Settings panel.
# 
# You can add collaborators to kernels for working as a team. When viewing the kernel (but not editting) you should see an "Access" button in the header. If you click on this you can add other Kaggle users as collaborators and give them view or edit access. You can add collaborators if the kernel is private or public.

# ## Wrapping up!
# 
# I hope you enjoyed this brief tutorial. Of course there is more to learn about using Kaggle's kernels, but that will come with experience. From here, try creating a kernel from one of the many [datasets](https://www.kaggle.com/datasets) or learning about machine learning in [one of our micro-courses](https://www.kaggle.com/learn/overview).
