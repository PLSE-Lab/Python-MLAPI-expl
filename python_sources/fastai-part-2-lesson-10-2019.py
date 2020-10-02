#!/usr/bin/env python
# coding: utf-8

# ## This is last part of fastai part 1 lesson 9, though it is mixed with my notes, eksperiments and what I found usefull, if you want the pure version, check fastai github or the following link: https://github.com/fastai/course-v3/tree/master/nbs/dl2

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2dne dejdeosnd djd sj')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import torch
import matplotlib.pyplot as plt


# # Callbacks
# ## Callbacks as GUI events

# In[ ]:


import ipywidgets as widgets #is to make buttoms and stuff


# In[ ]:


def f(o): print('hi')


# * the button widget is used to handle mouse clicks. The on_click method of the Button can be used to register function to be called when the button is clicked

# In[ ]:


w = widgets.Button(description='Click me')


# In[ ]:


w


# In[ ]:


w.on_click(f)


# 
# NB: When callbacks are used in this way they are often called "events".
# 
# Did you know what you can create interactive apps in Jupyter with these widgets? Here's an example from plotly:

# ## Creating your own callback

# In[ ]:


from time import sleep


# In[ ]:


def slow_calculation():
    res = 0
    for i in range(5):
        res += i*i
        sleep(1) #wait a secound 
    return res


# In[ ]:


slow_calculation() #so it will take 5 seound to mak the calulation 


# In[ ]:


def slow_calculation(cb=None): #cb=None callback we accept that it can take a parameter like funciton and let call it cb 
    res = 0
    for i in range(5):
        res += i*i
        sleep(1) #wait a sec.
        if cb: cb(i) #if there is a callback (cb) call it and pass in i #it could be the epoch number 
    return res


# In[ ]:


def show_progress(epoch):
    print(f"Awesome! We've finished epoch {epoch}!")


# In[ ]:


#so we take one function and pass it into another
slow_calculation(show_progress) #it count from 0 to 4 since the res+=i*i dosent influence i only range does that in slow_calculation
#do also note that it takes 5 sec. 


# ## Lambdas and partials

# In[ ]:


slow_calculation() #lambda o: print(f"Awesome! We've finished epoch {o}!")


# In[ ]:


def show_progress(exclamation, epoch):
    print(f"{exclamation}! We've finished epoch {epoch}!")


# In[ ]:


#but since we above wrote a function we only used ones, we can rewrite it and use lambda
slow_calculation(lambda o: show_progress("OK I guess", o)) # #lambda notation is just another way of whriting a function but we only uses it one
#so insted of 'def' we say 'lambda' and insted of parentheses we write the argument (o) and then what we want it to do. 
#note here that 'show_progress("OK I guess"' is the exclamation and 'o 'is epoch in the show_progress function above. 


# In[ ]:


#so lets say we want to make a function just takes exclamation we can do the below and we dont want to write the lambda in slow_calculation

def make_show_progress(exclamation):
    _inner = lambda epoch: print(f"{exclamation}! We've finished epoch {epoch}!") 
    return _inner


# In[ ]:


slow_calculation(make_show_progress("Nice!"))


# In[ ]:



def make_show_progress(exclamation):
    # Leading "_" is generally understood to be "private"
    def _inner(epoch): print(f"{exclamation}! We've finished epoch {epoch}!")
    return _inner


# In[ ]:


slow_calculation(make_show_progress("Nice!"))


# In[ ]:


f2 = make_show_progress("Terrific")


# In[ ]:


slow_calculation(f2)


# In[ ]:


slow_calculation(make_show_progress("Amazing"))


# It is so normal we want to only wanno give a function one parameter but 2 are regured so 
# we can use partial to solve this problem 

# In[ ]:


from functools import partial


# In[ ]:


slow_calculation(partial(show_progress, "OK I guess")) #so here we just pass in the function show_progress
#as the one parameter and then use "OK I guess" as an argument for the next parameter  
#so this now return a new function that just takes one parameter where the secound parameter is always given 


# In[ ]:


f3 = partial(show_progress, "OK I guess")


# In[ ]:


# f3() #now this function just takes one parameter which is epoch. since show_progress took to parameters one for epoch and the 
#secound was exclamation. but now exclamation is alwas "OK I guess". so the f2 only take the one parameter (epoch)


# In[ ]:


f3(1)


# ## Callbacks as callable classes

# In[ ]:


class ProgressShowingCallback():
    def __init__(self, exclamation="Awesome"): self.exclamation = exclamation #same a last lecture but we just store 
        #the exclamation value in a function
    def __call__(self, epoch): print(f"{self.exclamation}! We've finished epoch {epoch}!") #__call__ taking a objeckt(class-->ProgressShowingCallback) 
        #and treat it as if it was a function 


# In[ ]:


cb = ProgressShowingCallback("Just super")


# In[ ]:


#so we can call ProgressShowingCallback as if it was a function with paratenthess
# cb('hi')


# In[ ]:


slow_calculation(cb)


# ## Multiple callback funcs; *args and **kwargs

# In[ ]:


def f(*args, **kwargs): print(f"args: {args}; kwargs: {kwargs}")


# So every thing pass as a positional arguments (number, string) will *args turns it into a tuble and **kwargs turns the keyword arguments (ten=10, nine=9) into a dictionary 

# In[ ]:


f(3, 'a', thing1="hello")


# In[ ]:


f(3, 'a','b',9, thing1="hello", nine=9) #do remember as it is right here, the position of which you pass the argument er importen


# NB: We've been guilty of over-using kwargs in fastai - it's very convenient for the developer, but is annoying for the end-user unless care is taken to ensure docs show all kwargs too. kwargs can also hide bugs (because it might not tell you about a typo in a param name). In R there's a very similar issue (R uses ... for the same thing), and matplotlib uses kwargs a lot too.

# In[ ]:


def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb: cb.before_calc(i) #i we use it in PrintStatusCallback but not in PrintStepCallback
        res += i*i
        sleep(1)
        if cb: cb.after_calc(i, val=res) #i, val=res we use them in PrintStatusCallback but not in PrintStepCallback
    return res


# In[ ]:


class PrintStepCallback():
    def __init__(self): pass
    def before_calc(self, *args, **kwargs): print(f"About to start") #even though we dont use the arguments given from slow_calculation
        # we still have to make place for them and we do that with *args, **kwargs. 
    def after_calc (self, *args, **kwargs): print(f"Done step")


# In[ ]:


slow_calculation(PrintStepCallback())


# In[ ]:


class PrintStatusCallback():
    def __init__(self): pass
    def before_calc(self, epoch, **kwargs): print(f"About to start: {epoch}") #here we add **kwargs to make sure the code doesnt break if another argument are add to the function in the fucture
    def after_calc (self, epoch, val, **kwargs): print(f"After {epoch}: {val}") #but note we still use the arguments given from the function (slow_calculation)


# In[ ]:


slow_calculation(PrintStatusCallback())


# ## Modifying behavior

# the next thing we want write callback and to change something 

# In[ ]:


#early stopping 
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb and hasattr(cb,'before_calc'): cb.before_calc(i) #check if 'before_calc' exsist and call it if it is
        res += i*i
        sleep(1)
        if cb and hasattr(cb,'after_calc'): #chack if there is callback(cb) called 'after_calc' and only call it is it is 
            if cb.after_calc(i, res): #check the return value 
                print("stopping early") #and do something based on the returned value 
                break
    return res


# In[ ]:


class PrintAfterCallback():
    def after_calc (self, epoch, val):
        print(f"After {epoch}: {val}")
        if val>10: return True #cancel if our loop if the 'val' is greater then 10  


# In[ ]:


slow_calculation(PrintAfterCallback()) #and here we see that it stops at 14, since 14 is greater then 10 


# In[ ]:


#change the calulation 
class SlowCalculator():
    def __init__(self, cb=None): self.cb,self.res = cb,0 #defines what we need 
    
    #to use calc(function in this class) in ModifyingCallback we have to make the below function 
    def callback(self, cb_name, *args): #here u can also use __call__ and in the calc function u can just use self insted of self.callback
        if not self.cb: return #check to see if the given callback is defined 
        cb = getattr(self.cb,cb_name, None) #if it is it will grap it ...
        if cb: return cb(self, *args) #... and pass it into the calulator object itself (self) 
    
    #so we take our calulation function and putting it into a class so now the value it is calulation(res) is a attribute of the class 
    def calc(self):
        for i in range(5):
            self.callback('before_calc', i)
            self.res += i*i
            sleep(1)
            if self.callback('after_calc', i):
                print("stopping early")
                break


# In[ ]:


class ModifyingCallback():
    def after_calc (self, calc, epoch): #note the calculator (calc) functions calls on this funtions 
        print(f"After {epoch}: {calc.res}")
        if calc.res>10: return True #so we can now go into the calulator function and stop if the result gets greater then 10
        if calc.res<3: calc.res = calc.res*2 #and we can double the result by multipying with 2 if it is less then 3 


# In[ ]:


calculator = SlowCalculator(ModifyingCallback()) #for the changes from ModifyingCallback to be valid we pass it into the SlowCalculator class


# In[ ]:


calculator.calc() #have to call it like it is a class
calculator.res


# ## __dunder__ thingies

# In[ ]:


# __dunder__ thingles


# Anything that looks like 
# 
# 
#     __this__ 
# is, in some way, special. Python, or some library, can define some functions that they will call at certain documented times. For instance, when your class is setting up a new object, python will call 
# 
#     __init__
# These are defined as part of the python data model.
# 
# For instance, if python sees +, then it will call the special method 
# 
#     __add__
# If you try to display an object in Jupyter (or lots of other places in Python) it will call 
# 
#     __repr__

# In[ ]:


#exsampel
class SloppyAdder():
    def __init__(self,o): self.o=o #construck o 
    def __add__(self,b): return SloppyAdder(self.o + b.o + 0.01) #add o with b + 0.01 since it is sloppy
    def __repr__(self): return str(self.o)#printing 


# In[ ]:


a = SloppyAdder(1)
b = SloppyAdder(2)
a+b


# 
# Special methods you should probably know about (see data model link above) are:
# 
#      __getitem__
#      __getattr__
#      __setattr__
#      __del__
#      __init__
#      __new__
#      __enter__
#      __exit__
#      __len__
#      __repr__
#      __str__

# ## Variance and stuff
# ### Variance
# Variance is the average of how far away each data point is from the mean. E.g.:

# In[ ]:


t = torch.tensor([1.,2.,4.,18])


# In[ ]:


m = t.mean(); m


# In[ ]:


(t-m).mean()


# 
# Oops. We can't do that. Because by definition the positives and negatives cancel out. So we can fix that in one of (at least) two ways:

# In[ ]:


(t-m).pow(2).mean() #taking the power of 2 to the difference for a veribel and the mean(m) to the mean of it all 


# In[ ]:


(t-m).abs().mean() #taking the absolut value of the  difference for a veribel and the mean(m) to the mean of it all 


# 
# But the first of these is now a totally different scale, since we squared. So let's undo that at the end.

# In[ ]:


(t-m).pow(2).mean().sqrt() #we have to take the "kvadratrod" to get to the right scale again


# So the to good methods to find variance are
# 
#     (t-m).abs().mean()  #called mean absolut diviation 
# 
# and 
# 
#     (t-m).pow(2).mean().sqrt() #this is also called the standard diviation (std)
# 

# 
# They're still different. Why?
# 
# Note that we have one outlier (18). In the version where we square everything, it makes that much bigger than everything else.
# 
# (t-m).pow(2).mean() is refered to as variance. It's a measure of how spread out the data is, and is particularly sensitive to outliers.
# 
# When we take the sqrt of the variance, we get the standard deviation. Since it's on the same kind of scale as the original data, it's generally more interpretable. However, since sqrt(1)==1, it doesn't much matter which we use when talking about unit variance for initializing neural nets.
# 
# (t-m).abs().mean() is referred to as the mean absolute deviation. It isn't used nearly as much as it deserves to be, because mathematicians don't like how awkward it is to work with. But that shouldn't stop us, because we have computers and stuff.
# 
# Here's a useful thing to note about variance:

# In[ ]:


(t-m).pow(2).mean(), (t*t).mean() - (m*m)


#     (t*t).mean() - (m*m) #this is the math formular from below. And we want to use this veriation since it only go through the dataset ones while the other (traditunal) method goes through the dataset twice 

# You can see why these are equal if you want to work thru the algebra. Or not.
# 
# But, what's important here is that the latter is generally much easier to work with. In particular, you only have to track two things: the sum of the data, and the sum of squares of the data. Whereas in the first form you actually have to go thru all the data twice (once to calculate the mean, once to calculate the differences).
# 
# Let's go steal the LaTeX from Wikipedia:
# 
# $$\operatorname{E}\left[X^2 \right] - \operatorname{E}[X]^2$$
# 
# ## Covariance and correlation
# Here's how Wikipedia defines covariance:
# 
# $$\operatorname{cov}(X,Y) = \operatorname{E}{\big[(X - \operatorname{E}[X])(Y - \operatorname{E}[Y])\big]}$$

# In[ ]:


t #we use same data from variance 


# Let's see that in code. So now we need two vectors.

# In[ ]:


#exsampel
# `u` is twice `t`, plus a bit of randomness
u = t*2 #multiply t with 2 and set it = to u
u *= torch.randn_like(t)/10+0.95 #and put in some random noise 

plt.scatter(t, u); #plot


# In[ ]:


prod = (t-t.mean())*(u-u.mean()); prod #so now let compare the difference from 't' and its mean. With the difference in u and its mean and let multiply them together


# In[ ]:


prod.mean() #and let take the mean of that 


# In[ ]:


#so now lets take some random in a new verible and call it 'v'
v = torch.randn_like(t)
plt.scatter(t, v); #plot 't' and 'v'


# In[ ]:


((t-t.mean())*(v-v.mean())).mean() #and let us calulate the same product as before and take the mean of it 
#we see that this new number is much smaller then before (tensor(105.0522))


# the reason there is a big difference from the to coveriance is because in the first case it was all positive numbers in a line, so when we add them together they just give a big positive number. But in secound case the numbers was random and also choud be neagive, so this will give a very different result.

# 
# It's generally more conveniently defined like so:
# 
# $$\operatorname{E}\left[X Y\right] - \operatorname{E}\left[X\right] \operatorname{E}\left[Y\right]$$

# In[ ]:


cov = (t*v).mean() - t.mean()*v.mean(); cov


# 
# From now on, you're not allowed to look at an equation (or especially type it in LaTeX) without also typing it in Python and actually calculating some values. Ideally, you should also plot some values.
# 
# Finally, here is the Pearson correlation coefficient:
# 
# $$\rho_{X,Y}= \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}$$

# In[ ]:


cov / (t.std() * v.std())


# It's just a scaled version of the same thing. Question: Why is it scaled by standard deviation, and not by variance or mean or something else?

# ## Softmax
# Here's our final logsoftmax definition:

# In[ ]:


def log_softmax(x): return x - x.exp().sum(-1,keepdim=True).log()


# 
# which is:
# 
# $$\hbox{logsoftmax(x)}_{i} = x_{i} - \log \sum_{j} e^{x_{j}}$$
# And our cross entropy loss is:$$-\log(p_{i})$$
# 
# 

# so softmax 
# ![image.png](attachment:image.png)

# so output is just the last activation for an image here. 
# the exp is just the 'e' to the power of the last activation. in the buttom is the sum af exp at 12.70
# so softmax is just 'e' to the power of the last activation divided by the sum of all exp. 
# exsampel: 
# cat has a last activiation on 0.002 in this image 
# exp = e^0.002 (EXP(0.002))= 1.02
# softmax = 1.02/12.70 = 0.08 
# 

# so softmax is not good for images with more object in it. Or if there is non of the object you are looking for. 
# many use softmax anyhow since imagenet has tough us to do so. but non the less mistaken (imagenet has a picture with one object in it, so here is softmax good) since there in real life there can be more then one object in each picture. so softmax will nomatter what make a prediction and that there fx is a fish in the picture with high confidens even if its a picture of the color yellow. 

# insted of using softmax when there are more object or non-objects in pictures. here we should use binomial.
# Since better for this case. it check fx how much fish there is in the picture and this can be none or very close to none 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
# 
#      1.02/(1-1.02)

# In[ ]:


import numpy
#n : int or array_like of ints
#p : float or array_like of floats

def binomial(n,p):numpy.random.binomial(n, p, size=None)
    


# ## Browsing source code
# 
# 
# * Jump to tag/symbol by with (with completions)
# * Jump to current tag
# * Jump to library tags
# * Go back
# * Search
# * Outlining / folding
