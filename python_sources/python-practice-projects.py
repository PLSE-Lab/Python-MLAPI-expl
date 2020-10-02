# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print ("this is a test to print")

"""
This program generates passages that are generated in a cool mad-lib format
Author: Johan
"""

# The template for the story is below

STORY = "This morning %s woke up feeling %s. 'It is going to be a %s day!' Outside, a bunch of %s were protesting to keep %s in stores. They began to %s to the rhythm of the %s, which made all the %ss very %s. Concerned, %s texted %s, who flew %s to %s and dropped %s in a puddle of frozen %s. %s woke up in the year %s, in a world where %ss ruled the world."

print ("Mad Libs has started!")

name = raw_input("Enter a Name: ")
adj1 = raw_input("Enter an adjective: ")
adj2 = raw_input("Enter a second adjective: ")
adj3 = raw_input("Enter a third adjective: ")

verb = raw_input("Enter a verb: ")

noun1 = raw_input("Enter a noun: ")
noun2 = raw_input("Enter another noun: ")

animal = raw_input("Enter an animal: ")
food = raw_input("Enter a food: ")
fruit = raw_input ("Enter a fruit: ")
superhero = raw_input ("Enter a superhero: ")
country = raw_input ("Enter a country: ")
dessert = raw_input ("Enter a dessert: ")
year = raw_input ("Enter a year: ")

print (STORY % (name, adj1, adj2, animal, food, verb, noun1, fruit, adj3, name, superhero, name, country, name, dessert, name, year, noun2))

















