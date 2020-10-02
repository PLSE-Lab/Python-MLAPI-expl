#!/usr/bin/env python
# coding: utf-8

# In[1]:


def check_prime(number):
    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False
    return True

class Primes:
    def __init__(self, max):
        # the maximum number of primes we want generated
        self.max = max
        # start with this number to check if it is a prime.
        self.number = 1
        # No of primes generated yet. We want to StopIteration when it reaches max
        self.primes_generated = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.number += 1
        if self.primes_generated >= self.max:
            raise StopIteration
        elif check_prime(self.number):
            self.primes_generated+=1
            return self.number
        else:
            return self.__next__()


# In[2]:


prime_generator =Primes(10)


# In[3]:


for x in prime_generator:
    print(x)


# In[4]:


def Primes(max):
    number = 1
    generated = 0
    while generated < max:
        number += 1
        if check_prime(number):
            generated+=1
            yield number


# In[5]:


prime_generator = Primes(10)
for x in prime_generator:
    print(x)


# In[10]:


primes = (i for i in range(1,10) if check_prime(i))


# In[11]:



for x in primes:
    print(x)


# ## Pythagorean Triplets

# In[21]:


def triplet(n): # Find all the Pythagorean triplets between 1 and n
    for a in range(n):
        for b in range(a):
            for c in range(b):
                if a*a == b*b + c*c:
                    yield(a, b, c)


# In[22]:


triplet_generator = triplet(100)


# In[23]:


for x in triplet_generator:
    print(x)


# In[16]:


triplet_generator = ((a,b,c) for a in range(100) for b in range(a) for c in range(b) if a*a == b*b + c*c)


# In[17]:


for x in triplet_generator:
    print(x)


# In[ ]:




