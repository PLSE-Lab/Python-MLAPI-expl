#!/usr/bin/env python
# coding: utf-8

# # The goal of this tutorial is to introduce Object-oriented programing using a simple example.

# Import some libraries. 

# ![5.png](attachment:5.png)
# picture source: https://towardsdatascience.com/how-a-simple-mix-of-object-oriented-programming-can-sharpen-your-deep-learning-prototype-19893bd969bd

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#     We start a class (class in green) above and initialized it (__init__) by passing two attributes (name and a constant).
#     Note that we begin the __init__ with self.
#     

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2


#     Now we can introduce the first candidate (instances) into the class. we call it case_1. 
#     The case_1 has two attributes (name and a constant) as defined in the class.

# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)


#     Can we look inside the case_1 and see the attributes? let's try to print the case_1.

# In[ ]:


print(case_1)


#     As it is seen the python return case_1 (instance) as an object. 
#     To print the details in case_1, which is an object use the following codes.

# In[ ]:


print('name is = ',case_1.name)
print('cosntant_1 value is =', case_1.cons_1)
print('cosntant_2 value is =', case_1.cons_2)


#     We can ask the class to calculate a new attribute. 
#  

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2


#     In this case, we need to introduce the case_1 again.

# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)


# In[ ]:


print(case_1.new_attribute)


#     Let's define a method. The method is a function defined inside the class. 
#     The method can be introduced and executed as following.

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2
        
    def Method_name_1(self, x1, x2):
        '''You may name the method as you wish.
        this method takes two input parameters, two direct introduced values (x1, x2) and
        it uses two already introduced values (cons1 and cons2)
        and returns a value'''
        return (x1+x2)/(self.cons_1+self.cons_2)


#     Now that we do have a method inside the class, let's call it and see the output.
#     Again introduce the case_1 and ...

# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)


#     Execute the case_1 method using the following line. 
#     As it is seen two inputs are needed plus two constants were introduced as case_1 attributed already (above).

# In[ ]:


case_1.Method_name_1(1.2,5.4)


#     Now let's introduce one method inside another one.
#     This would be helpful when you do have large functions and dependants.
#     It makes your codes more clear and the chance of error will be reduced.

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2
        
    def Method_name_1(self, x1, x2):
        '''You may name the method as you wish.
        this method takes two input parameters, two direct introduced values (x1, x2) and
        it uses two already introduced values (cons1 and cons2)
        and returns a value'''
        return (x1+x2)/(self.cons_1+self.cons_2)
    
    def Method_name_2(self, x1, x2, x3, x4):
        '''You may name the method as you wish.
        this method takes four input parameters.'''
        
        OutPut_Method_name_1 = self.Method_name_1(x1, x2)
        
        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4
        
        return OutPut_Method_name_2


# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
case_1.Method_name_2(1.2, 5.4, 1, 10)


#     Now let's introduce case_2 with new attributes.
# 

# In[ ]:


case_2 = OOP_test1('name_as_you_wish', 6.5, 4)
case_2.Method_name_2(1.2, 5.4, 1, 10)


#     Let's look inside case_1 and case_2 and remind attributes and values.

# In[ ]:


print('case_1 attributes =', case_1.__dict__)
print('case_2 attributes =', case_2.__dict__)


#     We executed the Method_name_2 for both case 1 and 2, however, we do not see any details (No OutPut_Method_name_2 value is seen).
#     Let's add self inside the Method_name_2 and pass the OutPut_Method_name_2 value).

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2
        
    def Method_name_1(self, x1, x2):
        '''You may name the method as you wish.
        this method takes two input parameters, two direct introduced values (x1, x2) and
        it uses two already introduced values (cons1 and cons2)
        and returns a value'''
        return (x1+x2)/(self.cons_1+self.cons_2)
    
    def Method_name_2(self, x1, x2, x3, x4):
        '''You may name the method as you wish.
        this method takes four input parameters.'''
        
        OutPut_Method_name_1 = self.Method_name_1(x1, x2)
        
        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4
        
        self.OutPut_Method_name_2 = OutPut_Method_name_2
        
        return OutPut_Method_name_2


# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
case_1.Method_name_2(1.2, 5.4, 1, 10)
case_2 = OOP_test1('name_as_you_wish', 6.5, 4)
case_2.Method_name_2(1.2, 5.4, 1, 10)
print('case_1 attributes =', case_1.__dict__)
print('case_2 attributes =', case_2.__dict__)


#     Now we can see the output of Method_name_2.

#     Let's add Plot method to our class and execute it for each case.

# In[ ]:


class OOP_test1:
        
    def __init__(self, name, constant_1, constant_2):
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2
        
    def Method_name_1(self, x1, x2):
        '''You may name the method as you wish.
        this method takes two input parameters, two direct introduced values (x1, x2) and
        it uses two already introduced values (cons1 and cons2)
        and returns a value'''
        return (x1+x2)/(self.cons_1+self.cons_2)
    
    def Method_name_2(self, x1, x2, x3, x4):
        '''You may name the method as you wish.
        this method takes four input parameters.'''
        
        OutPut_Method_name_1 = self.Method_name_1(x1, x2)
        
        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4
        
        self.OutPut_Method_name_2 = OutPut_Method_name_2
        
        return OutPut_Method_name_2
    
    def Plot(self, x1, x2):
        
        
        self.x_values = [self.cons_1, self.cons_2]
        self.y_values = [x1, x2]
        
        plt.plot(self.x_values,self.y_values,'b^', markersize=4)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        
        


# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
case_2 = OOP_test1('name_as_you_wish', 6.5, 4)


# In[ ]:


case_1.Plot(2,5)


# In[ ]:


case_2.Plot(2,2)


#     What if we have 20 different cases and we like to plot all in the same figure.
#     In this case, we need to use the class method. 
#     First, introduce an empty instance and save all instances inside it.
#     Next, we used the class method and plot all instances (which are case_1 and case_2 in this tutorial).
#     Let's add some more lines to our class and see how does it work.

# In[ ]:


class OOP_test1:
    
    
    instances = [] # This line was added.
        
    def __init__(self, name, constant_1, constant_2):
        
        self.__class__.instances.append(self) # This line was added.
        
        self.name = name
        self.cons_1 = constant_1
        self.cons_2 = constant_2
        
        self.new_attribute = self.cons_1 + self.cons_2
        
    def Method_name_1(self, x1, x2):
        '''You may name the method as you wish.
        this method takes two input parameters, two direct introduced values (x1, x2) and
        it uses two already introduced values (cons1 and cons2)
        and returns a value'''
        return (x1+x2)/(self.cons_1+self.cons_2)
    
    def Method_name_2(self, x1, x2, x3, x4):
        '''You may name the method as you wish.
        this method takes four input parameters.'''
        
        OutPut_Method_name_1 = self.Method_name_1(x1, x2)
        
        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4
        
        self.OutPut_Method_name_2 = OutPut_Method_name_2
        
        return OutPut_Method_name_2
    
    def Plot(self, x1, x2):
        
        
        self.x_values = [self.cons_1, self.cons_2]
        self.y_values = [x1, x2]
        
        plt.plot(self.x_values,self.y_values,'b^', markersize=4)
        plt.xlabel('X')
        plt.ylabel('Y')
    
    
    @classmethod
    def Plot_whole_instances(cls):
        i = 1
        for instance in cls.instances:
            
            plt.subplot(1, 2, i)
            plt.plot(instance.cons_1,instance.cons_2,'bo', markersize=4)
            i += 1


# In[ ]:


case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
case_2 = OOP_test1('name_as_you_wish', 6.5, 4)


# In[ ]:


OOP_test1.Plot_whole_instances()


# I hope this tutorial be helpful. Comments are appreciated.
