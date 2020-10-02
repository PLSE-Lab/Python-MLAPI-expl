# This graph seeks to compare/predict ABV by IBU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main(path):
    beers = pd.read_csv(path)

    # Filters out null
    beers = beers[pd.notnull(beers.abv)]
    beers = beers[pd.notnull(beers.ibu)]

    beers_x = beers.abv[:,np.newaxis]
    beer_x_train = beers_x[:-100]
    beer_x_test = beers_x[-100:]

    beer_y_train = beers.ibu[:-100]
    beer_y_test = beers.ibu[-100:]

    regr = LinearRegression()

    regr.fit(beer_x_train, beer_y_train)

    # Plots
    plt.scatter(beer_x_test, beer_y_test, color="black")
    plt.plot(beer_x_test, regr.predict(beer_x_test), color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.title("ABV by IBU")

    plt.show()



if __name__ == "__main__":
    main("../input/beers.csv")
