#!/usr/bin/env python
# coding: utf-8

# what we learned in Python Data science toolbox section([here](https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners)) is to be reviewed in this homework

# ## User Def'd Functions, Tuple, String.Format, Scope

# In[ ]:


def getAreaAndPerimeter(a,b):
    return ( a*b ,2*(a+b))

w=4
h=8
a,p=getAreaAndPerimeter(w,h)
print("width: {0}, height: {1}, area: {2}, perimeter: {3}".format(h,w,a,p))


# In[ ]:


# global scope
xasd=5

def someFunc():
    #someFunc scope starts here
    xasd = 7
    print(xasd)
    #someFunc scope ends here
    
def anotherFunc():
    #anotherFunc scope starts here
    xasd = 9
    return xasd
    #anotherFunc scope ends here
    
#global scope here too
print(xasd)
someFunc()
print(anotherFunc())


# ## NESTED FUNCTION

# In[ ]:


def calculate(a,b,op):
    def add(a,b):
        return a+b
    def substract(a,b):
        return a-b
    
    if op == '+':
        return add(a,b)
    elif op == '-':
        return substract(a,b)
    elif op=='*' or op=='/':
        return 'not supported yet'
    else:
        return 'invalid op'
        
print(calculate(4,5,'+'))
print(calculate(7,5,'-'))
print(calculate(7,5,'*'))
print(calculate(7,5,'Q'))


# ## FLEXIBLE AND OPTIONAL ARGS

# In[ ]:


def calculate(*args,op='+'):
    def add(args):
        total=0
        for i in args:
            total = total + i
        return total

    if op == '+':
        return add(args)

print(calculate(4,5,6,7,op='+'))
print(calculate(4,5,6,7))


# In[ ]:


# print(calculate(4,5,6,7,'+')) 
# will raise error. opt args must be stated explicitly if used with flexible args


# ## LAMBDA FUNC

# In[ ]:


getSquare = lambda x : x * x
print(getSquare(4))


# ## ITERATOR

# In[ ]:


it = iter("doruk ulucay")
print(next(it))
print(next(it))
print(*it)


# ## ZIP

# In[ ]:


list1 = [1,3,5,7]
list2 = [2,4,6,8]
z = zip(list1, list2)
print(z)
z_list = list(z)
print(z_list)


unzipped =zip(*z_list)
l1, l2 = list(unzipped)
print(l1)


# ## LIST COMPREHENSION

# In[ ]:


list1 = [1,2,3]
list2 = [i + 1 for i in list1]
print(list1)
print(list2)

print([str(i) +' is even' if i%2==0 else str(i) +' is odd' for i in range(1,100) ])

