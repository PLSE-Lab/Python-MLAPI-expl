#
# This script transforms the 'ingredients' feature of the original dataset
# into separated columns. The values of these new columns are True or False,
# depending if the ingredient was used on the receipe.
#
# The ingredients need to be cleaned, making 'low fat mozzarella' and 
# 'reduced fat mozzarella' the same ingredient. Ideas are welcome.
# 

import pandas as pd
import matplotlib.pyplot as plt
#from nltk.corpus import stopwords
from itertools import chain

# Reading the data
train = pd.read_json('../input/train.json')

# Auxiliar function for cleaning
#cachedStopWords = stopwords.words("english")
def clean_ingredient(ingredient):
    # To lowercase
    ingredient = str.lower(ingredient)

    # Remove some special characters
    ingredient = ingredient.replace('&', '').replace('(', '').replace(')','')
    ingredient = ingredient.replace('\'', '').replace('\\', '').replace(',','')
    ingredient = ingredient.replace('.', '').replace('%', '').replace('/','')

    # Remove digits
    ingredient = ''.join([i for i in ingredient if not i.isdigit()])
    
    # Remove stop words
    #ingredient = ' '.join([i for i in ingredient.split(' ') if i not in cachedStopWords])

    return ingredient  

# The individual ingredients are stored in a set, avoiding duplicates
all_ingredients = {clean_ingredient(ingredient) for ingredient in chain(*train.ingredients)}

# fill the dataset with a column per ingredient
for ingredient in all_ingredients:
    train[ingredient] = train.ingredients.apply(lambda x: ingredient in x)

# Lets take a serie with the number of times each ingredient was used
s = train[list(all_ingredients)].apply(pd.value_counts).fillna(0).transpose()[True]

# Finally, plot the 10 most used ingredients
plt.style.use(u'ggplot')
fig = s.sort(inplace=False, ascending=False)[:10].plot(kind='barh')
fig = fig.get_figure()
fig.tight_layout()
fig.savefig('10_most_used_ingredients.jpg')