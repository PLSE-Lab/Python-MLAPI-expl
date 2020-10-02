#!/usr/bin/env python
# coding: utf-8

# <h1> Challenge Exercise with TensorFlow </h1>
# 
# <h1>Halley's Method</h1>
# 
# 
# 
# Use TensorFlow to find the roots of a fourth-degree polynomial using [Halley's Method](https://en.wikipedia.org/wiki/Halley%27s_method).  The five coefficients (i.e. $a_0$ to $a_4$) of 
# <p>
# $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
# <p>
# will be fed into the program, as will the initial guess $x_0$. Your program will start from that initial guess and then iterate one step using the formula:
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/142614c0378a1d61cb623c1352bf85b6b7bc4397" />
# <p>
# If you got the above easily, try iterating indefinitely until the change between $x_n$ and $x_{n+1}$ is less than some specified tolerance. Hint: Use [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)

# In[ ]:


import tensorflow as tf
import numpy as np


# <h2>$f(x)$</h2>

# In[ ]:


def fx(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    result=(a0  +tf.multiply(a1,tf.math.pow(x,1))
                +tf.multiply(a2,tf.math.pow(x,2))
                +tf.multiply(a3,tf.math.pow(x,3))
                +tf.multiply(a4,tf.math.pow(x,4))
             )
    return result


# <h2>$f'(x)$</h2>

# In[ ]:




def fxd(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    
    result=(a1  +2.0*tf.multiply(a2,tf.math.pow(x,1))
                +3.0*tf.multiply(a3,tf.math.pow(x,2))
                +4.0*tf.multiply(a4,tf.math.pow(x,3))
             )
    return result
    
  
    
    


# <h2>$f''(x)$</h2>

# In[ ]:


def fxdd(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    
    result=(2.0*a2
                +6.0*tf.multiply(a3,tf.math.pow(x,1))
                +12.0*tf.multiply(a4,tf.math.pow(x,2))
             )
    return result


# <h2>$h(x)$</h2>

# In[ ]:



def hx(coeffs,x):
    result=x-(
                (2.0*fx(coeffs,x)*fxd(coeffs,x))
                /
                (
                    2.0*fxd(coeffs,x)*fxd(coeffs,x)
                    -fxd(coeffs,x)*fxdd(coeffs,x)
                )
            )
    return result
    


# 
# $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
# <p>
#  Trying to test for 2nd degree equestion
#  $a_0=-2,
#    a_1=0,
#    a_2=1 ,
#    a_3=0,
#    a_4 =0,
#    $
#    <p>
# $f(x) = -2 +   x^2 $
#     
#    

# In[ ]:


coeffs = tf.placeholder(dtype=tf.float64, shape=(5,))
x = tf.placeholder(dtype=tf.float64)
finaloutcome = hx(coeffs, x)
with tf.Session() as sess:
    a=[-2.0,0.0,1.0,0.0,0.0]
    r=sess.run(finaloutcome,feed_dict={coeffs:a,x:2})
    print(r)
    for i in range(10):
        r=sess.run(finaloutcome,feed_dict={coeffs:a,x:r})
        print(r)


# $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
# <p>
#  Trying to test for 3nd degree equestion
#  $a_0=-27,
#    a_1=0,
#    a_2=0 ,
#    a_3=1,
#    a_4 =0,
#    $
#    <p>
# $f(x) = -27 +   x^3$

# In[ ]:


coeffs = tf.placeholder(dtype=tf.float64, shape=(5,))
x = tf.placeholder(dtype=tf.float64)
finaloutcome = hx(coeffs, x)
with tf.Session() as sess:
    a=[-27.0,0.0,0.0,1.0,0.0]
    r=sess.run(finaloutcome,feed_dict={coeffs:a,x:2})
    print(r)
    for i in range(10):
        r=sess.run(finaloutcome,feed_dict={coeffs:a,x:r})
        print(r)


# In[ ]:




