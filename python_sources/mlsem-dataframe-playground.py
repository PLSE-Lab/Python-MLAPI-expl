
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def dataframe_playground():
    imdbData = pd.read_csv("../input/IMDB-Movie-Data.csv")
    imdbData = imdbData.dropna(axis=0, how='any')

    print("The input was loaded into a " + str(type(imdbData)))
    listOfColumnNames = imdbData.columns.values
    print(listOfColumnNames)

    print(imdbData.info())
    print(imdbData.describe())
    #print('data head\n' + str(imdbData.head(3)))

    sns.distplot(imdbData['Rating'])

    ratingAndMetaScore = pd.concat([imdbData['Rating'], imdbData['Metascore']], axis=1)
    ratingAndMetaScore.plot.scatter(x='Rating', y='Metascore')
    #https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    #https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python

    print("Skewness: %f" % imdbData['Rating'].skew())
    print("Kurtosis: %f" % imdbData['Rating'].kurt())


    highestRatedMovies = imdbData.sort_values(by=['Rating'], ascending=False)
    print("\n\nThe highest rated movies are:\n")
    print(highestRatedMovies[['Rating', 'Title', 'Revenue (Millions)']].head(5))

    moviesWithHighestRevenue = imdbData.sort_values(by=['Revenue (Millions)'], ascending=False)
    print("\n\nThe movies with the highest revenue are:\n")
    print(moviesWithHighestRevenue[['Rating', 'Title', 'Revenue (Millions)']].head(5))

    ratingRevenuePairs = imdbData[['Rating', 'Revenue (Millions)']].values
    ratingArray = list([element[0] for element in ratingRevenuePairs])
    revenueArray = list([element[1] for element in ratingRevenuePairs])
    plt.figure(3)
    plt.plot(ratingArray, revenueArray, 'go')
    plt.xlabel('Rating')
    plt.ylabel('Revenue')
    plt.title('Revenue vs rating plot')

    imdbData["RatingOverRevenue"] = imdbData['Rating'] / imdbData['Revenue (Millions)']
    print("=-=-=-=-=-=-=-")
    print("=-=-=-=-=-=-=-")
    print("=-=-=-=-=-=-=-")
    print(imdbData[["RatingOverRevenue", 'Rating', 'Revenue (Millions)', 'Title']].head(15))

    imdbData.hist(column='Revenue (Millions)', bins=30)

    imdbData['IHaveAlreadySeenIt'] = np.random.choice([True, False], imdbData.shape[0])
    imdbData['MyFriendHasAlreadySeenIt'] = np.random.choice([True, False], imdbData.shape[0])

    bestMoviesThatWeHaventSeenYet = imdbData.loc[(imdbData['IHaveAlreadySeenIt'] == False) & (imdbData['MyFriendHasAlreadySeenIt'] == False), ['Title', 'Rating']].sort_values(by='Rating', ascending=False).head(5)
    print("=-=-=-=-=-=-=-")
    print("=-=-=-=-=-=-=-")
    print("=-=-=-=-=-=-=-")
    print("The best movies that we havent seen yet:")
    print(bestMoviesThatWeHaventSeenYet)
    plt.show()