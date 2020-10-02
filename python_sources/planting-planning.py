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
import random
def planting_planner(sq_ft):
    """takes an input of square feet of planting space and zip_code.
    returns number and type of seeds to plant in a percentage of total square feet
    plus the dates to start them for them to germinate successfully in a given zip_code.
    could use pandas in the future to generate a planting diagram.
    >>> planting_planner(100, 60450)

    60 sq_ft corn.
    30 sq_ft potatoes.
    10 sq_ft Peanuts"""

    three_nums=[]
    #60 percent of squarefeet
    c_c_c=['Wheat','Cereal Rye','Oats','Barley','Triticale','Corn','Sorghum','Amaranth','Quinoa','Mature Fava Beans','Sunflowers','Filberts','Raisins']
    #30 percent of sq_feet
    root_crops=['Potatoes','Jerusalem Artichoke','Garlic','Leeks','Parsnips','Sweet Potatoes','Salsify']
    #10 percent of sq_feet
    cash_crops=['Peanuts','Soybeans','Beans','Burdock','Cassava','Onions','Turnips','Rutabaga']
    three_nums.append(random.randint(0,len(c_c_c)-1))
    three_nums.append(random.randint(0,len(root_crops)-1))
    three_nums.append(random.randint(0,len(cash_crops)-1))
    return [c_c_c[three_nums[0]],int(sq_ft*.60),"start date",root_crops[three_nums[1]],int(sq_ft*.3),"start date",cash_crops[three_nums[2]],int(sq_ft*.1),"start date"]
planting_planner(100)