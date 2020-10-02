#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# ---
# 

# In this exercise, you will write your first lines of code and learn how to use the coding environment for the micro-course!
# 
# ## Setup
# 
# First, you'll learn how to run code, and we'll start with the code cell below.  (Remember that a **code cell** in a notebook is just a gray box containing code that we'd like to run.)
# - Begin by clicking inside the code cell.  
# - Click on the blue triangle (in the shape of a "Play button") that appears to the left of the code cell.
# - If your code was run sucessfully, you will see `Setup Complete` as output below the cell.
# 
# ![ex0_run_code](https://i.imgur.com/TOk6Ot4.png)

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# The code cell above imports and configures the Python libraries that you need to complete the exercise.
# 
# Now, follow the same process to run the code cell below.  If successful, you'll see `Setup Complete` as output.

# In[46]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex1 import *
print("Setup Complete")


# The code that you just ran sets up the system that will give you feedback on your work.  You'll learn more about the feedback system in the next step.
# 
# ## Step 1: Explore the feedback system
# 
# In order to successfully finish this micro-course, you'll need to complete various hands-on coding exercises.  Each  exercise allows you to put your new skills to work with a real-world dataset.  Along the way, you'll receive feedback on your work.  We'll tell you if an answer you've written is correct or incorrect, provide customized hints, and show you our official solution (_if you'd like to take a look!_).
# 
# To explore the feedback system, we'll start with a simple example of a coding problem.  Follow the following steps in order:
# 1. Begin by running the code cell below without making any edits.  This should return the following output: 
# > <font color='#ccaa33'>Check:</font> When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `one`
# 
#     This feedback tells us that there are some necessary changes to the code that we haven't made yet: we need to set the variable `one` to something other than the blank provided below (`____`).  
# 
# 
# 2. Replace the underline with a value of `2`, so that the line of code appears as `one = 2`.  Then, run the code cell.  This should return the following output:
# > <font color='#cc3333'>Incorrect:</font> Incorrect value for `one`: `2`
# 
#     This feedback tells us that the value that we've provided is incorrect: `2` is not the correct answer here!
# 
# 
# 3. Now, change the value of `2` to `1`, so that the line of code appears as `one = 1`.  Then, run the code cell.  The answer should be marked as <font color='#33cc33'>Correct</font>, and, you have now completed this problem!
# 
# 
# In this exercise, you are responsible for filling in the line of code that sets the value of variable `one`.  **Please never edit the code that is used to check your answer.**  So, lines of code like `step_1.check()` and `step_2.check()` should always be left as provided.

# In[30]:


# Fill in the line below
one = 1

# Check your answer
step_1.check()


# This problem was relatively straightforward, but for more difficult problems, you may like to receive a hint or view the official solution.  Run the code cell below now to receive both for this problem.

# In[31]:


step_1.hint()
step_1.solution()


# ## Step 2: Load the data
# 
# Now, we're ready to get started with some data visualization code!  You'll begin by loading the dataset from the previous tutorial.  
# 
# Recall that loading a dataset into a notebook is done in two parts:
# - begin by specifying the location (or [filepath](https://bit.ly/1lWCX7s)) where the dataset can be accessed, and then
# - use the filepath to load the contents of the dataset into the notebook.
# 
# We have provided the first part for you, and you need to fill in the second part to set the value of `fifa_data`.  Feel free to copy this code from the tutorial.  Once running the code returns a <font color='#33cc33'>Correct</font> result, you're ready to move on!

# In[32]:


# Path of the file to read
fifa_filepath = "../input/fifa.csv"

# Fill in the line below to read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True )

# Check your answer
step_2.check()


# Recall the difference between comments and executable code:
# - **Comments** are preceded by a pound sign (`#`) and contain text that appear faded and italicized.  They are completely ignored by the computer when the code is run.
# - **Executable code** is code that is run by the computer.
# 
# In the code cell below, every line is a comment:
# ```python
# # Uncomment the line below to receive a hint
# #step_2.hint()
# #step_2.solution()
# ```
# 
# If you run the code cell (that appears below this big block of text) as-is, it won't return any output.  Try this now!
# 
# Next, remove the pound sign before `step_2.hint()` so that the code cell appears as follows:
# ```python
# # Uncomment the line below to receive a hint
# step_2.hint()
# #step_2.solution()
# ```
# When we remove the pound sign before a line of code, we say we **uncomment** the line.  This turns the comment into a line of executable code that is run by the computer.  Run the code cell now, which should return the <font color='#3366cc'>Hint</font> as output.
# 
# Finally, uncomment the line to see the solution, so the code cell appears as follows:
# ```python
# # Uncomment the line below to receive a hint
# step_2.hint()
# step_2.solution()
# ```
# Then, run the code cell.  You should receive both a <font color='#3366cc'>Hint</font> and the <font color='#33cc99'>Solution</font>.
# 
# If at any point you're having trouble with coming up with the correct answer to a problem, you are welcome to obtain either a hint or the solution before completing the cell.  (So, you don't need to get a <font color='#33cc33'>Correct</font> result before running the code that gives you a <font color='#3366cc'>Hint</font> or the <font color='#33cc99'>Solution</font>.)

# In[33]:


# Uncomment the line below to receive a hint
step_2.hint()
# Uncomment the line below to see the solution
step_2.solution()


# ## Step 3: Review the data
# 
# In the next code cell, use a Python command to print the first 5 rows of the data.  Please completely erase the underline (`____`) and fill in your own code.   
# 
# If you don't remember how to do this, please take a look at the previous tutorial, ask for a <font color='#3366cc'>Hint</font>, or view the <font color='#33cc99'>Solution</font>.  The code you write here won't give you feedback on whether your answer is correct, but you'll know if your answer is right if it prints the first 5 rows of the dataset!

# In[34]:


# Print the last five rows of the data 
fifa_data.head(5) # Your code here


# Use the first 5 rows of the data to answer the question below.

# In[48]:


# Fill in the line below: What was Brazil's ranking (Code: BRA) on December 23, 1993?
brazil_rank = 3.0

# Check your answer
step_3.check()


# If you haven't already, uncomment the lines and run the code to view the <font color='#3366cc'>Hint</font> and the <font color='#33cc99'>Solution</font>.

# In[39]:


# Lines below will give you a hint or solution code
#step_3.hint()
#step_3.solution()


# ## Step 4: Plot the data
# 
# Now that the data is loaded into the notebook, you're ready to visualize it!  
# 
# #### Part A
# 
# Copy the code from the tutorial that we used to make a line chart.  This code may not make sense just yet - you'll learn all about it in the next tutorial!
# 
# Before proceeding to **Part B** of this question, make sure that you get a <font color='#33cc33'>Correct</font> result.  If you need help, feel free to view the <font color='#3366cc'>Hint</font> or the <font color='#33cc99'>Solution</font>.

# In[45]:


# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Fill in the line below: Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data) # Your code here

# Check your answer
step_4.a.check()


# If you haven't already, uncomment the lines and run the code to view the <font color='#3366cc'>Hint</font> and the <font color='#33cc99'>Solution</font>.

# In[ ]:


# Lines below will give you a hint or solution code
#step_4.a.hint()
#step_4.a.solution_plot()


# #### Part B
# 
# Some of the questions that you'll encounter won't require you to write any code.  Instead, you'll generally need to interpret visualizations.  These questions don't require a <font color='#33cc33'>Correct</font> result, and you won't be able to check your answer.  
# 
# However, you can receive a <font color='#3366cc'>Hint</font> to guide the way you think about the question, or you can view our official <font color='#33cc99'>Solution</font>.  
# 
# As an example, consider the question: Considering only the years represented in the dataset, which countries spent at least 5 consecutive years in the #1 ranked spot?
# 
# To receive a <font color='#3366cc'>Hint</font>, uncomment the line below, and run the code cell.

# In[49]:


step_4.b.hint()


# To see the <font color='#33cc99'>Solution</font>, uncomment the line and run the code cell.

# In[50]:


step_4.b.solution()


# Congratulations - you have completed your first coding exercise!
# 
# # Keep going
# 
# Move on to learn to create your own **[line charts](https://www.kaggle.com/alexisbcook/line-charts)** with a new dataset.

# ---
# **[Data Visualization: From Non-Coder to Coder Micro-Course Home Page](https://www.kaggle.com/learn/data-visualization-from-non-coder-to-coder)**
# 
# 
