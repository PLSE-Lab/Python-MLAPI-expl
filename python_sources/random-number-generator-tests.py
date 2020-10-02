# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt


# Set the number of primes you want to find.
x = 10

# The First Example; goes through a list of numbers if the number is prime it gets added to the
# prime list. The more primes you try to find, they slower this runs, because as prime numbers 
# become larger they also become further apart.

print('Running Example 1')

# Start the timer for the test.
Start_Example1 = dt.datetime.now()

# Start the prime list.
p = [2]

# Incrementally count integers.
n = 1

# Setup a test list for numbers to go in while it's being determined if they're prime.
t = []


while len(p) < x:
    # Can add 2 every time and skip all of the even values, since they are all divisible by 2
    # and thus are not prime.
    n = n+2
    # For n, check every prime in the list, p.
    for i in p:
        t.append(n%i)
    # If there are any primes in the list that divide exactly into n, then throw n out and move
    # to the next odd number.
    if 0 in t:
        t = []
        continue
    # If there aren't any primes in the list that divide exactly into n, then it must be a prime.
    # So, add it to p.
    else:
        p.append(n)
        t = []
        

End_Example1 = dt.datetime.now()

print('Run Time '+str(End_Example1-Start_Example1))
                
    

