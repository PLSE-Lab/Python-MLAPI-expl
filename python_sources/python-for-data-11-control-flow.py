#!/usr/bin/env python
# coding: utf-8

# # Python for Data 11: Control Flow
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Although Python is a popular tool for data analysis, it is a general-purpose programming language that wasn't designed specifically for that task. It is important to know some basic Python programming constructs so that you can write custom code and functions to handle situations where built in functions and libraries fall short.
# 
# When you run code in Python, each statement is executed in the order in which they appear. Programming languages like Python let you change the order in which code executes, allowing you to skip statements or run certain statements over and over again. Programming constructs that let you alter the order in which code executes are known as control flow statements.

# ## If, Else and Elif

# The most basic control flow statement in Python is the "if" statement. An if statement checks whether some logical expression evaluates to true or false and then executes a code block if the expression is true.
# 
# In Python, an if statement starts with if, followed by a logical expression and a colon. The code to execute if the logical expression is true appears on the next line, indented from the if statement above it by 4 spaces:

# In[1]:


x = 10                # Assign some variables
y = 5

if x > y:             # If statement
    print("x is greater than y")


# In the code above, the logical expression was true--x is greater than y--so the print(x) statement was executed.
# 
# If statements are often accompanied by else statements. Else statements come after if statements and execute code in the event that logical expression checked by an if statement is false:

# In[2]:


y = 25
x = 10

if x > y:
    print("x is greater than y")
else:
    print("y is greater than x")


# In this case the logical expression after the if statement is false, so the print statement after the if block is skipped and the print statement after the else block is executed instead.
# 
# You can extend this basic if/else construct to perform multiple logical checks in a row by adding one or more "elif" (else if) statements between the opening if and closing else. Each elif statement performs an additional logical check and executes its code if the check is true:

# In[3]:


y = 10

if x > y:
    print("x is greater than y")
elif x == y:
    print("x and y are equal!")
else:
    print("y is greater than x")


# ## For Loops

# For loops are a programming construct that let you go through each item in a sequence and then perform some operation on each one. For instance, you could use a for loop to go through all the values in a list, tuple, dictionary or series and check whether each conforms to some logical expression or print the value to the console.
# 
# Create a for loop using the following syntax:

# In[4]:


my_sequence = list(range(0,101,10))    # Make a new list

for number in my_sequence:  # Create a new for loop over the specified items
    print(number)           # Code to execute


# In each iteration of the loop, the variable "number" takes on the value of the next item in the sequence.
# 
# For loops support a few special keywords that help you control the flow of the loop: continue and break.
# 
# The continue keyword causes a for loop to skip the current iteration and go to the next one:

# In[5]:


for number in my_sequence:
    if number < 50:
        continue              # Skip numbers less than 50
    print(number)             


# The "break" keyword halts the execution of a for loop entirely. Use break to "break out" of a loop:

# In[6]:


for number in my_sequence:
    if number > 50:
        break              # Break out of the loop if number > 50
    print(number)     


# In the for loop above, substituting the "continue" keyword for break would actually result in the exact same output but the code would take longer to run because it would still go through each number in the list instead of breaking out of the for loop early. It is best to break out of loops early if possible to reduce execution time.

# ## While Loops

# While loops are similar to for loops in that they allow you to execute code over and over again. For loops execute their contents, at most, a number of iterations equal to the length of the sequence you are looping over. While loops, on the other hand, keep executing their contents as long as a logical expression you supply remains true:

# In[7]:


x = 5
iters = 0

while iters < x:      # Execute the contents as long as iters < x
    print("Study")
    iters = iters+1   # Increment iters by 1 each time the loop executes


# While loops can get you into trouble because they keep executing until the logical statement provided is false. If you supply a logical statement that will never become false and don't provide a way to break out of the while loop, it will run forever. For instance, if the while loop above didn't include the statement incrementing the value of iters by 1, the logical statement would never become false and the code would run forever. Infinite while loops are a common cause of program crashes.
# 
# The continue and break statements work inside while loops just like they do in for loops. You can use the break statement to escape a while loop even if the logical expression you supplied is true. Consider the following while loop:

# In[8]:


while True:            # True is always true!
    print("Study")
    break              # But we break out of the loop here


# It is important to make sure while loops contain a logical expression that will eventually be false or a break statement that will eventually be executed to avoid infinite loops.
# 
# Although you can use a while loop to do anything a for loop can do, it is best to use for loops whenever you want to perform a specific number of operations, such as when running some code on each item in a sequence. While loops should be reserved for cases where you don't know how many times you will need to execute a loop.

# ## The np.where() Function

# Although it is important to be able to create your own if/else statements and loops when you need to, numpy's vectorized nature means you can often avoid using such programming constructs. Whenever you want to perform the same operation to each object in a numpy or pandas data structure, there's often a way to do it efficiently without writing your own loops and if statements.
# 
# For example, imagine you have a sequence of numbers and you want to set all the negative values in the sequence to zero. One way to do it is to use a for loop with an inner if statement:

# In[9]:


import numpy as np

# Draw 25 random numbers from -1 to 1
my_data = np.random.uniform(-1,1,25)  

for index, number in enumerate(my_data):  
    if number < 0:               
        my_data[index] = 0            # Set numbers less than 0 to 0

print(my_data)


# *Note: "enumerate" takes a sequence and turns it into a sequence of (index, value) tuples; enumerate() lets you loop over the items in a sequence while also having access the item's index.*
# 
# Using a for loop to perform this sort of operation requires writing quite a bit of code and for loops are not particularly fast because they have to operate on each item in a sequence one at a time.
# 
# Numpy includes a function called where() that lets you perform an if/else check on a sequence with less code:

# In[10]:


my_data = np.random.uniform(-1,1,25)  # Generate new random numbers

my_data = np.where(my_data < 0,       # A logical test
                   0,                 # Value to set if the test is true
                   my_data)           # Value to set if the test is false

print(my_data)


# Not only is np.where() more concise than a for loop, it is also much more computationally efficient.

# ## Wrap Up

# Control flow statements are the basic building blocks of computer programs. Python and its libraries offer vast number functions, but general-use functions can't apply to every situation. Sooner or later, you'll need to write a custom code to perform a task unique to your specific project or data set. Next time we'll learn how to package control flow statements into reusable functions.

# ## Next Lesson: [Python for Data 12: Functions](https://www.kaggle.com/hamelg/python-for-data-12-functions)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
