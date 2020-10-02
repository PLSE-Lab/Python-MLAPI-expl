# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def general_statistics(data):
    print("For all beers collected:")
    print("The lowest ABV:{0}. The average ABV: {1}. The highest ABV: {2}.".format(data['abv'].min(), data['abv'].mean(), data['abv'].max()))
    print("The lowest IBU:{0}. The average IBU: {1}. The highest IBU: {2}.".format(data['ibu'].min(), data['ibu'].mean(), data['ibu'].max()))


def main(path):
    dataset1 = pd.read_csv(path)
    dataset1['floatabv'] = dataset1['abv'].astype(float)
    
    general_statistics(dataset1)

if __name__ == "__main__":
    main("../input/beers.csv")