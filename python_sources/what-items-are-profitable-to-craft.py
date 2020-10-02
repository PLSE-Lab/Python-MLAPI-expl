#!/usr/bin/env python
# coding: utf-8

# # Figuring out the most profitable recipes
# 
# In Animal Crossing: New Horizons, you can sell the items you collect for bells (in game currency) or craft them into new items. Since I like to min-max the way I play my games, I wanted to know if there were any craftables that sell for more than their parts.
# 
# ## 1. Inspecting the Data
# We're going to import the "recipes" table, and slice it so that we have the recipe name, the materials list, and the sell price of the item.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

recipes = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/recipes.csv')
rp = recipes.iloc[:,0:15]
rp.head(5)


# Note that we do have the Buy column, but our calculations will basically ignore it for the purposes of this exercise.
# 
# ## 2. Writing the materials function
# Now let's write a function that returns the material list and sell price for a given recipe.

# In[ ]:


def get_materials(item_name):
    # Slice the table for the item we want.
    material_list = rp[rp['Name'] == item_name]
    
    # Initialize dictionary and iterator
    material_dict = {}
    i = 0
    
    # Iterate over the material and quantity columns
    # The material name is our key, the quantity is our value to that key
    while i < 6:
        material_dict[material_list.iloc[0,i*2+2]] = material_list.iloc[0,i*2+1]
        i += 1
    
    # Get rid of the nan value keys since we don't need them.
    if np.nan in material_dict:
        material_dict.pop(np.nan)
    
    # Clean our 'Sell' value so that it is an integer, not an array.
    material_dict['Sell'] = material_list['Sell'].values.sum()
    return material_dict

print(get_materials('tire toy'))
print(get_materials('barrel'))


# As you can see, this function will return a dictionary containing the materials and sell prices of our recipe items.
# 
# ## 3. Return difference between sell price and materials price
# Now we need to make a function that returns the difference between the recipe sell price with the sum of the sell prices of the materials. If the result is positive, then it is more profitable to sell the crafted recipe than to sell all the materials separately and vicaversa.

# In[ ]:


# Import the data for the raw materials
mat = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/other.csv')

# Function for calculating difference between sell price and sell price of all materials in recipe.
def materials_diff(item_name):
    # Run the get_materials function to create dictionary
    mat_dict = get_materials(item_name)
    
    # Initialize sell prices
    mat_price = 0
    sell_price = 0
    
    for key, val in mat_dict.items():
        # The sell price becomes the same as sell_price (integer only)
        if key == 'Sell':
            sell_price = val.sum()
        
        # The materials in our dictionary are matched to the Sell prices in our raw materials table.
        # Material prices are multiplied by the quantity (values) from the dictionary to get total raw material sell price.
        else:
            material_cost = mat[mat["Name"] == key]['Sell'].values.sum()
            mat_price = mat_price + material_cost * val
    return sell_price-mat_price

print(materials_diff('tire toy'))
print(materials_diff('barrel'))


# It looks like the tire toy might be an item we would want to craft to sell. The barrel, not so much.
# 
# ## 4. Putting this into our table
# Now we'll use list comprehension to iterate over a list of recipe names and calculate the craft-to-sell profit for each recipe.

# In[ ]:


# Generate recipe name list
name_list = list(rp.Name)

# Iterate over recipe name list to create a list of craft-to-sell profit margins.
craft_profit = [materials_diff(x) for x in name_list]

# Make this list a column and check table
rp['craft_profit'] = craft_profit
rp.head(5)


# In[ ]:


# Make a new table only containing profitable recipes.
profitable_recipes = rp[rp['craft_profit'] > 0].sort_values(by=['craft_profit'], ascending=False)
profitable_recipes


# Now that we have this list, let's consider some limitations:
# * If the materials for a recipe were not in the "other.csv" file, the craft-to-sell profit will simply equal the sell price of the recipe.
# * Even if we included all item data here, the function does not do a good job of accounting for items crafted from items (such as wooden-block furniture). It can only show if it is more or less profitable to sell a given recipe after crafting it or immediately before assuming all materials are present.
# 
# While understanding that we likely have some skewed data, let's just check the profit distribution from our new item list.

# In[ ]:


sns.distplot(profitable_recipes['craft_profit'], bins=10)
plt.show()


# OK, so it looks like there's many items that sell for very little profit and many that sell for more (but probably should not be included in the data).
# 
# Lets compare this to insects and fish, which are far more common as sellable items.

# In[ ]:


ins = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/insects.csv')
fish = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/fish.csv')
sns.distplot(ins['Sell'], bins=10)
sns.distplot(fish['Sell'], bins=10)
plt.xlim(0, 25000)
plt.show()


# While there's a strong skew towards cheaper items, we can tell that the sell prices for fish and insects is, on average, considerably higher than our "profitable" craftables.
# 
# ## 5. Conclusion
# The normal ways of generating income (i.e. catching small creatures, harvesting fruit, playing the stalk market) are far more profitable than trying to use the DIY system to generate profit.
