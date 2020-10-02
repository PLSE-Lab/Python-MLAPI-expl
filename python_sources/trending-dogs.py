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
import matplotlib.pyplot as plt
# look at owners of most popular dog
def main():
    plt.style.use('ggplot')
    # Import the CSV into a dataframe
    df = pd.read_csv('../input/20151001hundehalter.csv');
    # Drop null rows
    df = df[pd.notnull(df['ALTER'])]
    # df = create_minage_maxage_cols(df);
    # print(top_ten_dogs(df))
    top_three = top_three_dogs(df)
    # print(top_dog_name, top_dog_freq)
    # print(top_three.index)
    top_dog_owners_bday_histogram(df, list(top_three.index.values))
# calculate top dog
def top_three_dogs(df):
    dog_types = df['RASSE1']
    top_dog = dog_types.value_counts()
    return top_dog[0:3]

def create_minage_maxage_cols(df):
    print('Creating Minage Maxage columns')
    # create age_min and age_max columns
    age_series = df['ALTER']
    age_min_list = [];
    age_max_list = [];
    for index, value in age_series.iteritems():
        minage,maxage = value.split("-")
        age_min_list.append(minage)
        age_max_list.append(maxage)
    # add age_min and age_max columns
    return df.assign(age_min=age_min_list, age_max=age_max_list)
# show some trends of the top dog
def top_dog_owners_bday_histogram(df, top_dog_names):
    # print(top_dog_names)
    # all rows where top dog
    top_dog_rows = df.loc[df['RASSE1'] == top_dog_names[0]]
    sec_dog_rows = df.loc[df['RASSE1'] == top_dog_names[1]]
    tri_dog_rows = df.loc[df['RASSE1'] == top_dog_names[2]]
    # histogram of Mischling Dog Birth dates
    top_dog_bdays = top_dog_rows['GEBURTSJAHR_HUND']
    sec_dog_bdays = sec_dog_rows['GEBURTSJAHR_HUND']
    tri_dog_bdays = tri_dog_rows['GEBURTSJAHR_HUND']
    plt.hist(top_dog_bdays, bins=range(1990, 2017 + 1, 1), label=top_dog_names[0],alpha=1.0, normed=True)
    plt.hist(sec_dog_bdays, bins=range(1990, 2017 + 1, 1), label=top_dog_names[1],alpha=0.5, normed=True)
    # plt.hist(tri_dog_bdays, bins=range(1990, 2017 + 1, 1), label=top_dog_names[2],alpha=0.75, normed=True)
    plt.title("Birthdates of Top Two Dogs in Zurich")
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()
if __name__ == "__main__":
    main()
