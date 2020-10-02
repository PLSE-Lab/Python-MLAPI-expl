# 3rd-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# assesses how close two strings are
from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# construct a map between a df key and its value categories count
def map_columnCount(mapDf_in, column_in):
    # a map between a platform name and its categories count
    categoriesCount = {}

    for key in mapDf_in:
        # count categories among all records
        categoriesCount[key] = mapDf_in[key].groupby(column_in).size()
        
    return categoriesCount

# construct a 2d-list whose elements represent maps sizes of some category

def map_categoriesSize(categories_in, mapDf_in, dfCategoriesCount_in):

    categoriesSize = {}

    # loop on categories
    for category in categories_in:
        categoryList = []
        # append platforms sizes of current category
        for key in mapDf_in:
            categoryList.append(dfCategoriesCount_in[key][category])
        # append platforms sizes of the category into whole 2d-list
        categoriesSize[category] = categoryList
        
    return categoriesSize

# construct categories size as 2d list in order of
# extremely high, very high, .., low

def ConstCategoriesSize_2dList(categoriesSize_in):

    categoriesSize_2dList = []

    # desired order of extremely high, very high, .., low
    categories_size_itemsReversed = reversed(list(categoriesSize_in.items()))

    # loop on categories names and their corresponding elements
    for category, elemsLis in categories_size_itemsReversed:
        # append to whole 2dList
        categoriesSize_2dList.append(elemsLis)

    # convert to numpy array
    categoriesSize_2dList = np.array(categoriesSize_2dList)
    
    return categoriesSize_2dList

def showGroupedBars(categoriesSize_in, groupsNames_in, yLabel_in, title_in):

    # for calculating x-position
    bars_count = len(categoriesSize_in)

    # labels initial locations
    x = np.arange(len(groupsNames_in))
    width = 0.1  # the width of the bars

    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()

    # translate canvas, documented for future works
    #translate(ax, 0, -(width*bars_count/2))

    # for each category, construct its corresponding bar
    for idx, category in enumerate(categoriesSize_in):
        ax.bar(x + (idx*width) - (width*bars_count/2), categoriesSize_in[category], width, label=category, align='edge')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel_in)
    ax.set_title(title_in)
    ax.set_xticks(x)
    ax.set_xticklabels(groupsNames_in)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
    
def showCategoricalHeatmap(ySize_in, xSize_in, categoriesSize_2dList_in, groupsNames_in, categoriesSize_in, title_in):

    # construct figure and axes
    fig = plt.figure(figsize=(ySize_in, xSize_in))
    ax = plt.subplot()

    im = ax.imshow(categoriesSize_2dList_in)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(groupsNames_in)))
    ax.set_yticks(np.arange(len(categoriesSize_in)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(groupsNames_in)
    ax.set_yticklabels(reversed(list(categoriesSize_in)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(categoriesSize_in)):
        for j in range(len(groupsNames_in)):
            text = ax.text(j, i, categoriesSize_2dList_in[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title_in)
    fig.tight_layout()
    plt.show()
    
def constPlatformsPairs(platform_df):
    # a map of platforms pairs to the two platforms distance dataframe
    platformsPairs_df = {}

    # loop on pairs of platforms
    for idx, platformName in enumerate(platform_df):
        for idx_, platformName_ in enumerate(platform_df):
            # compare only unique pairs
            if idx_ > idx:
                # get dataframe of the first platform
                df = platform_df[platformName]
                # get dataframe of the second platform
                df_ = platform_df[platformName_]

                # recall that we work on indices instead of repeating already stored data

                # first platform's indices
                firstColumn = df.index
                # second platform's indices
                secondColumn = df_.index

                # column name corresponding to the first platform
                firstColumnStr = platformName + "_index"
                # column name corresponding to the second platform
                secondColumnStr = platformName_ + "_index"

                # construct a dataframe of the cartesian product of two platforms indices
                # cartesian product is constructed as multiindex first
                index = pd.MultiIndex.from_product([firstColumn, secondColumn], names = [firstColumnStr, secondColumnStr])
                # move cartesian product indices to concrete columns data
                tem_df = pd.DataFrame(index = index).reset_index()

                # compute similarity between each pairs of the cartesian product corresponding game titles
                # in lambda, given the index, map it to the game's title, then finally apply similar function
                tem_df['similarity'] = tem_df.apply(lambda x: similar(df.loc[x[firstColumnStr], 'title'], df_.loc[x[secondColumnStr], 'title']), axis=1)

                # group according to left-hand-side game title
                # each group represents a comparison of one game from the first platform up against all other games from the second platform
                grouped = tem_df.groupby([firstColumnStr])
                # filter a group into records of maximum similarity. we expect only one is obtained
                # each group maps a game from the first platform to another game from the second platform whose similarity is maximum
                applied = grouped.apply(lambda g: g[g['similarity'] == g['similarity'].max()])
                # remove index resulted by groupby
                tem_df = applied.reset_index(drop=True)

                # store the dataframe by both platforms names
                platformsPairs_df[platformName + '_' + platformName_] = tem_df
                
    # discard any pairs whose similarity is not equal to 1
    # in our case, games names are exactly the same among platforms. However, that might not be always the case

    # for each pair of platforms
    for pair in platformsPairs_df:
        # filter only records whose similariy is equal to 1
        platformsPairs_df[pair] = platformsPairs_df[pair][platformsPairs_df[pair]['similarity'] == 1]            
    
    return platformsPairs_df

def constPlatformsPairs_diff(platformsNames, platformsPairs_df, platform_df):
    # a map of platforms pairs to dataframe showing computed results
    platformsPairs_diff = {}

    # loop on platforms pairs
    for idx, name in enumerate(platformsNames):
        for idx_, name_ in enumerate(platformsNames):
            # consider only unique pairs
            if idx_ > idx:
                # meaningful name of the pair which corresponds to other constructed dataframes
                pairStr = name + '_' + name_

                # get games titles from the first platform. same as obtaining them from the second platform
                # platform name is appended by _index corresponding platformsPairs_df. an index is returned
                # the returned index is mapped to the game's titles by platform_df
                # note that lambda is applied on each record
                title = platformsPairs_df[pairStr].apply(lambda x: platform_df[name].loc[x[name+"_index"], 'title'], axis=1)

                # compute distance of ratings of the two platforms
                # index is returned from platformsPairs_df, then mapped to the numeric value of rating
                # distance is computed by the two game's rating on the two platforms
                user_diff = platformsPairs_df[pairStr].apply(lambda x: abs(platform_df[name].loc[x[name+"_index"], 'user_rating'] - platform_df[name_].loc[x[name_+"_index"], 'user_rating']), axis=1)
                critic_diff = platformsPairs_df[pairStr].apply(lambda x: abs(platform_df[name].loc[x[name+"_index"], 'critic_rating'] - platform_df[name_].loc[x[name_+"_index"], 'critic_rating']), axis=1)
                userCritic_diff = platformsPairs_df[pairStr].apply(lambda x: abs(platform_df[name].loc[x[name+"_index"], 'userCritic_difference'] - platform_df[name_].loc[x[name_+"_index"], 'userCritic_difference']), axis=1)

                # construct a new dataframe of results just computed
                platformsPairs_diff[pairStr] = pd.DataFrame({
                    'title': title,
                    'user_diff': user_diff,
                    'critic_diff': critic_diff,
                    'userCritic_diff': userCritic_diff
                }
                )
                
    return platformsPairs_diff