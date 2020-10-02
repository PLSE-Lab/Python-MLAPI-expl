#!/usr/bin/env python
# coding: utf-8

# <center><font size="5">Courses Summary - Python</font></center>
# 

# <b>Summary</b><br>
# <a href='#1.1'>1.1 - Variable assignment, reassigning variable, function calls</a><br>
# <a href='#1.2'>1.2 - Type of variable</a><br>
# <a href='#1.3'>1.3 - Numbers and arithmetic</a><br>
# <a href='#1.4'>1.4 - Builtin functions for working with numbers</a><br>
# <a href='#2.1'>2.1 - Getting help</a><br>
# <a href='#2.2'>2.2 - Defining functions</a><br>
# <a href='#3.1'>3.1 - Booleans</a><br>
# <a href='#3.2'>3.2 - Comparison operations</a><br>
# <a href='#3.3'>3.3 - Combining boolean values</a><br>
# <a href='#3.4'>3.4 - Conditionals</a><br>
# <a href='#3.5'>3.5 - Boolean conversion</a><br>
# <a href='#3.6'>3.6 - Conditional expression (aka 'ternary')</a><br>

# <label id='1.1'><b>1.1 - Variable assignment, reassigning variable, function calls</b></label>

# In[ ]:


# Variable assignment
variable = 10

# Reassigning variable
variable = variable + 10

# Function calls
print(variable)


# <label id='1.2'><b>1.2 - Type of variable</b></label>

# In[ ]:


# Describe the type of "thing" that variable is
print(type(variable)) # int
print(type(19.95))    # float


# <label id='1.3'><b>1.3 - Numbers and arithmetic</b></label>

# <table style="width:50%">
#   <caption style="text-align: center;">Python Arithmetics Operators</caption>
#   <tr>
#     <th width="10%" style="text-align: center;">Operator</th>
#     <th width="30%" style="text-align: center;">Name</th> 
#     <th width="60%" style="text-align: left;">Description</th>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a + b</td>
#     <td style="text-align: center;">Addition</td> 
#     <td style="text-align: left;">Sum of a and b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a - b</td>
#     <td style="text-align: center;">Subtraction</td> 
#     <td style="text-align: left;">Difference of a and b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a * b</td>
#     <td style="text-align: center;">Multiplication</td> 
#     <td style="text-align: left;">Product of a and b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a / b</td>
#     <td style="text-align: center;">True division</td> 
#     <td style="text-align: left;">Quotient of a and b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a // b</td>
#     <td style="text-align: center;">Floor division</td> 
#     <td style="text-align: left;">Quotient of a and b, removing fractional parts</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a % b</td>
#     <td style="text-align: center;">Modulus</td> 
#     <td style="text-align: left;">Integer remainder after division of a by b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a ** b</td>
#     <td style="text-align: center;">Exponentiation</td> 
#     <td style="text-align: left;">a raised to the power of b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">-a</td>
#     <td style="text-align: center;">Negation</td> 
#     <td style="text-align: left;">The negative of a</td>
#   </tr>
# </table>
# <p style="text-align:center;">
#     Order of operations &rarr; PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)
# </p>
# 

# In[ ]:


# True division
print(5 / 2) # 2.5
print(6 / 2) # 3.0

# Floor division
print(5 // 2) # 2
print(6 // 2) # 3

# Modulus
print(5 % 2) # 1
print(6 % 2) # 0


# <label id='1.4'><b>1.4 - Builtin functions for working with numbers</b></label>

# In[ ]:


# Minimun value
print("Min:", min(1, 2, 3))  # 1

# Maximun value
print("Max:", max(1, 2, 3))  # 3

# Absolute value
print("Abs:", abs(-32))     # 32

# Cast to float and to integer
print("Float:",float(10))    # 10.0
print("Integer:",int(3.33))    # 3
print("Integer:",int("807"))   # 807


# <label id='2.1'><b>2.1 - Getting help</b></label>

# In[ ]:


help(print) # Common pitfall: pass in the name of the function itself, and not the result of calling.


# <label id='2.2'><b>2.2 - Defining functions</b></label>

# In[ ]:


def defining_function(a, b = None): # Expect obrigatory parameter and optional parameter
    """Docstring that is return when the user calls the help functiom    
    >>> defining_function(a)
    a
    """
    return a # Return is optional

# You can supply functions as arguments to other functions.
defining_function(defining_function(10))


# <label id='3.1'><b>3.1 - Booleans</b></label>

# In[ ]:


x, y = True, False
print(x,y)
print(type(x), type(y))


# <label id='3.2'><b>3.2 - Comparison operations</b></label>

# <table style="width:50%">
#   <caption style="text-align: center;">Python Comparison Operations</caption>
#   <tr>
#     <th width="15%" style="text-align: center;">Operation</th>
#     <th width="34%" style="text-align: left;">Description</th> 
#     <th width="2%"></th>
#     <th width="15%" style="text-align: center;">Operation</th> 
#     <th width="34%" style="text-align: left;">Description</th>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a == b</td>
#     <td style="text-align: left;">a equal to b</td> 
#     <td></td>
#     <td style="text-align: center;">a != b</td> 
#     <td style="text-align: left;">a not equal to b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a < b</td>
#     <td style="text-align: left;">a less than b</td> 
#     <td></td>
#     <td style="text-align: center;">a > b</td> 
#     <td style="text-align: left;">a greater than b</td>
#   </tr>
#   <tr>
#     <td style="text-align: center;">a  b</td>
#     <td style="text-align: left;">a less than or equal to b</td> 
#     <td></td>
#     <td style="text-align: center;">a >= b</td> 
#     <td style="text-align: left;">a greater than or equal to b</td>
#   </tr>
# </table>

# <label id='3.3'><b>3.3 - Combining boolean values</b></label>

# In[ ]:


print(True and True)  # and
print(True or False)  # or
print(not True)       # not
# Precedence: not, and, or


# <label id='3.4'><b>3.4 - Conditionals</b></label>

# In[ ]:


x = 2
if(x == 0):               # if
  print(x, "is zero")
elif(x < 1):
  print(x, "is negative") # elif
else:
  print(x, "is positive") # else


# <label id='3.5'><b>3.5 - Boolean conversion</b></label>

# In[ ]:


print(bool(3), 3)                                 # All numbers are treated as true, except 0 
print(bool(0), 0 )
print(bool("test"), "test")                       # All strings are treated as true, except the empty string ""
print(bool([]), "[]")                             # Generally empty sequences are "falsey" and the rest are "truthy"    
print(bool({'dict': 'value'}), {'dict': 'value'})


# <label id='3.6'><b>3.6 - Conditional expression (aka 'ternary')</b></label>

# In[ ]:


expression = True
variable = 'True' if expression else 'False'
print(variable)

