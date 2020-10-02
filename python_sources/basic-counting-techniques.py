#!/usr/bin/env python
# coding: utf-8

#  # Motivation (Data Science and Probaility, Statistics)
#  
#  Data Science is impossible without a solid knowledge of probability and statistics. But often we just lose our interest in this because of boring and complicated statistic terms. Yes, probability and statistics are boring, but today I want to change your attitude to this and show a little bit more convenient way to learn all the important concepts. When we are playing with a big amount of data where every second is important we have to use some formulas to get things done in minutes if we solve it without using maths it may take an year or more to solve the problem
#  
#  
# # Introduction
#  In this Notebook we will learn about some really simple and awful probability and statistics techniques. I followed the edx course (Fat Chance probability 0 to 1)
#  
#  **Title**
#  1. Counting
#      1. Simple Counting
#      2. Multiplication Principle
#      3. Subtration Principle

# # Simple Counting

# 
# # What if I say that tell me total Numbers between 1-200? Its 200 easy enough. 1--300 its 300 numbers. Easy Total Numbers Between (1-n) is n. But what If I say tell me numbers between 3344 and 4566. Emm now How can we calculate it in secoends. Including (3344 and 4566) in it 
# 
# 1. K = Starting Number
# 2. N = Larger Number 
# 
# So the formula will be which is (n-(k-1)) ---> n-k+1
# 
# ![counting.PNG](attachment:counting.PNG)

# In[ ]:


k = 5
n = 10
print("Total Numbers between {} and {} is {}".format(k,n,n-k+1))


# # What if we want to know that in list [101 ... 401] How many numbers are divisible by 5?
# 
# 1. Take first element in list which is divisible by 5 (in our case its 105)
# 2. Take Last element in list which is divisible by 5 (in our case its 400)
# 3. Apply formula 
# 
# # Formula
# > ( First_element + (n-1)5 = last_element) -----> n = (last_element - first_element / 5) + 1
# 
# # Time Difference:
# 
# You will see that their is a hugh time difference in fucntion in which we use some counting formula and on the other hand we using sequential method to solve the problem (like most of the programmers do)
# 

# In[ ]:


import time
def magic_math(k,n,divisor):
    '''
    Description
    This funtion will find out how many numbers from range k to n are divisible by every element in a list
    e.g 
        if k=10, n= 19999 and divisor = 3 than This funtion will find out the count of numbers
        in range 10 to 19999 which are divisible by 3
    
    '''
    # loop from (5 -- 10) taking first element which is divisible by 3
    first_divisible_element = [i for i in range(k,(k + divisor) + 1) if i % divisor == 0][0]  
    # loop from (95 -- 100) taking last element which is divisible by 3
    last_divisible_element = [i for i in range(n - divisor, n + 1) if i % divisor == 0][-1]
    return int(((last_divisible_element - first_divisible_element) / divisor) + 1)

def cross_check(k,n,divisor):
    return sum([1 for i in range(k,n+1) if i % divisor ==0])


# Starting Point
k = 3
# End Point
n = 100000000
# Divisor 
divisor = 3


import time
start_time = time.time()
magic_function = magic_math(k,n,divisor)
magic_funtion_time = (time.time() - start_time)

start_time = time.time()
cross_check = cross_check(k,n,divisor)
cross_check_time = (time.time() - start_time)
print("Magic Funtion {} Time Taken: {:.16f}".format(magic_function, magic_funtion_time))
print("Cross Check: {} Time Taken: {:.16f}".format(cross_check, cross_check_time ))



# # Now give the Length of Numbers from list which are divisible by 4,3,5.
# 
# I think now its difficult but no problem I will give you a better formula to do that. Its a basic formula that we are studing from 4 standard. 

# In[ ]:


import numpy as np
def cross_check_divisor(k, n, divisor_list):
    '''
    Description
    This funtion will find out how many numbers from range k to n are divisible by every element in a list
    e.g 
        if k=10, n= 19999 and divisor_list = [2,3,4,5] than This funtion will find out the count of numbers
        in range 10 to 19999 which are divisible by 2,3,4,5
    
    '''
    single_divisor_pass_count = 0
    all_divisors_pass_count = 0 
    for i in range(k, n+1):
        single_divisor_pass_count = 0
        for divisor in divisor_list:
            if i % divisor == 0:
                single_divisor_pass_count = single_divisor_pass_count + 1
        if single_divisor_pass_count == len(divisor_list): 
            all_divisors_pass_count = all_divisors_pass_count + 1
    return all_divisors_pass_count


# Start Point
k = 3
# End Point
n = 10000000
# Divisor List
divisor_list = [3,4,5]

start_time = time.time()
magic_function = magic_math(k,n,int(np.lcm.reduce(divisor_list)))
magic_funtion_time = (time.time() - start_time)

start_time = time.time()
cross_check = cross_check_divisor(k, n, divisor_list)
cross_check_time = (time.time() - start_time)

print("Magic Funtion {} Time Taken: {:.16f}".format(magic_function, magic_funtion_time))
print("Cross Check: {} Time Taken: {:.16f}".format(cross_check, cross_check_time ))


# Practice Examples:
# 
# # Question 1
# In a sports stadium with numbered seats, every seat is occupied except seats  33  through  97 .
# 
# How many seats are still available? Choose the best answer.
# 
# 1. 63 
# 2. 64 
# 3. 65 
# 4. 97 
# 
# Correct Answer : 65
# 
# # Question 2
# 
# In a sports stadium with numbered seats, every seat is occupied except seats  33  through  97 .
# 
# Suppose the fans are superstitious and only want to sit in even numbered seats because, otherwise, they fear their team will lose. How many even numbered seats are still available in the block of seats numbered  33  through  97 ? Choose the best answer.
# 1. 31 
# 2. 32 
# 3. 33 
# 4. 34
# 
# 
# 
# Correct Answer : 32

# # Multiplication Principle.
# Sometime we have to find out combinations that our algorithm have to find process and we also have to calculate that time taken by our algorithm to perform that action e.g if our algorithm will takes 1 sec to process 1 combination. than our algorithm will takes 9000sec to process 9000 combinations. i.e 150 mins to performa task. In some cases we have to find alot of combinations.
# 
# 
# # Real time Example
# I am going to share a real time example that I have been gone through.
# 
# I am working as data-scientist in programmers force. Now our goal is to make image clear enough to read text from the image for that we have to apply a list of filters to get our work done. We were working on real time and you know that every image is different from other because some images are capture with dslr some with varity of mobile cams now if we apply every filter on the image some images are crystal clear but some becomes so bad and if we are playing with images order of filters do matter e.g if we apply erision filter on image and than apply dillation filter on the image their will be no effect on the image so this combination is so bad. and if we apply erision filter and than apply remove_small_dots filter on the image this combination gives really satisfying results. Now we have to find combination of filters that we will apply on every image and predict it from our model and keep on apply filter until the model confidence becomes greater than 95 percent. We have total of 12 filters and in 1 request we have to process approximate 12 images. for every request our alogrithm is taking 100ms to process 1 combination on 1 image. Now we have total of **12!(479001600) combination** and **12 images**. So total we have **5748019200 combination to process 1 request**  approximatly 18 years to process 1 request. Damn if we go into the coding without knowing about the figures we lost our 1 to 2 weeks that will be totaly useless. So its really good to find out algorithm complexity before diving into the code and improve your algorithms on paper to save your time.
# 
# 
# In this section we will learn some multiplicative techniques that will help us out in finding our algorithm's time complexity

# In[ ]:


# What if i say that tell me number of words that which have possible 5 characters init..
# What will be the answer

# we have 26 alphabets 
# Every character in a word possibly includes a charater from 26 words.

print (26 * 26 * 26 * 26 * 26)  # 11881376 total words

# Now what if I say that no letter will repeat in the word. so on first place we have 26 possible characters
# on second place we have 25 possible charaters on third place we have 24 possible charaters and so on.

print(26 * 25 * 24 * 23 * 22) # 7893600 total words

# Now what if I say find out all combination of number of length 3 and number should not consists 0 init 
# i.e 111, 223, 332 etc Here we have have 9 possible
# digits to place at every position

print(9 * 9 * 9) # 729 number will be formed in length of 3

# Now what if I say find out all the numbers of length 3, number shouldnot consists 0 init and all the numbers
# will be odd (means ending with odd number) i.e 111,223 , 335 etc Try it yourself see what you got.

#Now for this we can take a different approach we will start from right to left. How many odd digits we have?
#(1,3,5,7,9) i.e 5 so __5 we have 5 possible values to fill on the right most side it ensure that it will be odd
#now on the middle place we have 8 numbers left (think it in your mind you will be amazed) and on the left place
#we have 7 digits left.

#so the answer will be 

print (7*8*5) # 280 odd numbers will be the list of 3 digits numbers

# these things are very simple but will help us out in solving really big problems


# # Subtraction Principle
# 
# Subtraction principle is some thing we can say that find out numbers of words consists of length 4 but have atleast 1 vowel init. Emm now here multiplicative principle fails so badly and subtraction principle come to rescue us. So what will be the answer.
# 
# we know that Number of words that consists of 4 letters are (26 * 26 * 26 * 26)  but how can we say compute that these words consists of atleast on vowel 
# 
# for example if we have
# 
# abc[26] if we have a b c word where a is a vowel we have 26 letter available for the last slot but
# bbs[5] in this case where we have no vowel on first 3 slots  we have only 5 choices so here is something that is dependent on the previous selection. SO what can we do here
# 
# 
# We know that we have 5 vowels in a english and 21 consonants and we can easily calculate words consists of only consonants i.e 21 * 21 * 21 * 21 i.e 194481 Now we can easily calculate words include atleast 1 vowels i.e
# 
# 
# Total words       - Words include only consonants = Words includes atleast 1 vowel
# 26 * 26 * 26 * 26 - 21 * 21 * 21 * 21             = 262495

# These are some examples of basic counting in probability. I hope you will like these examples and will be amazaed by viewing the time complexity reduced when we use maths formula. In the next notebook I will talk about 
# 
# # Advance Counting Techniques
# 
# which will be useful for you in handling data with the help of maths.
