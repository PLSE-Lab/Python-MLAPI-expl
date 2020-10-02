#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# generate pi to nth digit
# Chudnovsky algorihtm to find pi to n-th digit
# from https://en.wikipedia.org/wiki/Chudnovsky_algorithm
# https://github.com/microice333/Python-projects/blob/master/n_digit_pi.py

import decimal
def compute_pi(n):
    decimal.getcontext().prec = n + 1
    C = 426880 * decimal.Decimal(10005).sqrt()
    K = 6.
    M = 1.
    X = 1
    L = 13591409
    S = L
    for i in range(1, n):
        M = M * (K ** 3 - 16 * K) / ((i + 1) ** 3)
        L += 545140134
        X *= -262537412640768000
        S += decimal.Decimal(M * L) / X
    pi = C / S
    return pi


while True:
    n = int(input("Please type number between 0-1000: "))
    if n >= 0 and n <= 100000000000000000000000000000000000:
        break
print(compute_pi(n))


# In[ ]:





# In[ ]:


# Python program to display all the prime numbers within an interval

# change the values of lower and upper for a different result
lower = 99999999
upper = 9999999999999999999

# uncomment the following lines to take input from the user
#lower = int(input("Enter lower range: "))
#upper = int(input("Enter upper range: "))

print("Prime numbers between",lower,"and",upper,"are:")

for num in range(lower,upper + 1):
   # prime numbers are greater than 1
   if num > 1:
       for i in range(2,num):
           if (num % i) == 0:
               break
       else:
           print(num)

