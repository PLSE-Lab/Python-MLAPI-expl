#!/usr/bin/env python
# coding: utf-8

# # Foot Insole Validation Algorithm 
# 
# This is a prototype to demonstate how the foot insole validation simulation will work for brands who will go through the # Von homer certification process. 
# The methodology below is the first success criteria in order to move foward 
# ![AI](http://www.makeplain.com/wp-content/uploads/2015/05/machineLearning_iconDark.png)
# 

# In[ ]:


#Create object to contain input parameters of client (Brand) foot insole parameters
#combine dictionary parameters into one big json chunk (clear code)
insole = {"fw":int, "rfw":int, "fml":int, "fthml":int, "ah":int, "hhfm":int, "fl":int, 
                "isInsoleValid":True}


# In[ ]:


#Foot insole difference holder 
foot_insole_difference = {"fw":int, "rfw":int, "fml":int, "fthml":int, "ah":int, "hhfm":int, "fl":int}


# In[ ]:


#foot parameter key dictionary 
foot_parameter_key_dictionary = {"fw":"Foot Meterasel", "rfw":"Rear Foot Width", "fml":"First Metatarsel Length",
                                 "fthml":"Fifth Metarsel Length", "ah":"Arch Height",
                                 "hhfm":"Heel to head of First Metatarsel Phalangeal Joint",  "fl":"Foot Length"}


# In[ ]:


#Add data input gui here 
from ipywidgets import widgets
from IPython.display import display

#Brute Force Methodology Iterate dimensions in the futre
footData = widgets.Text(description="Foot Metatarsel")
footData2 = widgets.Text(description="Rear Foot Width")
footData3 = widgets.Text(description="First Metatarsel Length")
footData4 = widgets.Text(description="Fifth Metarsel Length")
footData5 = widgets.Text(description="Arch Height")
footData6 = widgets.Text(description="Heel to Head First Metatarsel Philangeal Joint")
footData7 = widgets.Text(description="Foot Length")
display(footData, footData2, footData3, footData4, footData5, footData6, footData7)
    
def handle_submit(sender):
    insole["fw"] = int(footData.value)
    
def handle_submit2(sender):
    insole["rfw"] = int(footData2.value)

def handle_submit3(sender):
    insole["fml"] = int(footData3.value)

def handle_submit4(sender):
    insole["fthml"] = int(footData4.value)
    
def handle_submit5(sender):
    insole["ah"] = int(footData5.value)

def handle_submit6(sender):
    insole["hhfm"] = int(footData6.value)
    
def handle_submit7(sender):
    insole["fl"] = int(footData7.value)
#complete.on_submit(handle_submit)
#print(insole)
footData.on_submit(handle_submit)
footData2.on_submit(handle_submit2)
footData3.on_submit(handle_submit3)
footData4.on_submit(handle_submit4)
footData5.on_submit(handle_submit5)
footData6.on_submit(handle_submit6)
footData7.on_submit(handle_submit7)


# In[ ]:


#Todo create upper and lower lengths to every foot variable 
lower_foot_ranges= {"fw":9, "rfw":7, "fml":6, "fthml":7, "ah":9, "hhfm":12, "fl":3}
upper_foot_ranges = {"fw":10, "rfw":12, "fml":15, "fthml":19, "ah":8, "hhfm":13, "fl":18}


# In[ ]:


#print(insole) Debug state
#computation for finding the parameters that need to be updated based on foot dimensions
def inSoleValidator(name,lowerLength, upperLength):
    #checks if number is outside of then it is in invalid
    if insole[name] < lowerLength or insole[name] > upperLength:
        if insole[name] < lowerLength:
            foot_insole_difference[name] = upperLength - insole[name] 
            print(foot_parameter_key_dictionary[name],"Add", foot_insole_difference[name])
        if insole[name] > lowerLength: 
            foot_insole_difference[name] = upperLength - insole[name]
            print(foot_parameter_key_dictionary[name],"Reduce", foot_insole_difference[name])
    #use case if the value is true or not
    else: print(foot_parameter_key_dictionary[name], "measurment is valid")

#iterates through each of the inputs, and runs it through the validator functions
for i in foot_insole_difference:
    lowerLength = lower_foot_ranges[i]
    upperLength = upper_foot_ranges[i]
    inSoleValidator(i,lowerLength, upperLength)
        
    

