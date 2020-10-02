import numpy as np
import matplotlib.pyplot as plt

def f(x): ##define fuction in terms of x
    y=.5*(x**2)
    return y


a=1. ##Lower Bound
b=3.0  #Upper Bound
N=10**6  #Maximum Number of Iterations

timedomain=np.arange(1,N+1,1)   #Number of Iterations
epsrange=np.zeros_like(timedomain,dtype=float) ##List of Variance with respect to Number of Iterations.
Integralrange=np.zeros_like(timedomain,dtype=float) ##List of Integral values with respect to Number of Iterations.

##initial variables
Sum=0.
eps=1
count=0
Integral=0


##Monte Carlo Run based off the Mean Value Theorem
while count<N:
    count+=1
    Integralold=Integral
    r=np.random.random_sample()*(b-a)+a
    Sum+=f(r)
    Integral=(b-a)*Sum/count
    eps=(abs(((Integral**2)-(Integralold)**2)**.5))
    epsrange[count-1]=eps
    Integralrange[count-1]=Integral



plt.plot(timedomain,epsrange,'ro',timedomain, Integralrange)
print("Integral =", Integralrange[-1], "Variance=", epsrange[-1])

    
        
        
        
    
    
    


