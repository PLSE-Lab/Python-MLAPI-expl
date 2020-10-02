# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

world_food_facts = pd.read_csv("../input/FoodFacts.csv", low_memory = False)
#print(world_food_facts.loc[:, [world_food_facts.countries, world_food_facts.allergens.notnull()]])
countries_and_allergies = world_food_facts[["countries", "allergens"]]
countries_and_allergies.allergens = world_food_facts.allergens.str.lower()
print(countries_and_allergies[countries_and_allergies.allergens.notnull()])
#print(world_food_facts[world_food_facts.allergens_en.notnull()].allergens_en.first())





# Any results you write to the current directory are saved as output.