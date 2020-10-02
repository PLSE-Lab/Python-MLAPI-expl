
#importing libraries

import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import sklearn.exceptions
from IPython.display import display, Image
from IPython.core.display import HTML
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# data extraction

# data extraction


def extraction(file):

    Img = Image(url="https://store-guides2.djicdn.com/guides/wp-content/uploads/2018/11/Banner_Buying-Guide.jpg")
    display(Img)
    pd.options.display.max_rows = 10
    pd.options.display.max_columns = 999

    # Data Extraction
    black = pd.read_csv(file)
    black = pd.DataFrame(black)

    black.info()
    # print(black.shape)
    return black.iloc[1:10]
    

def clean(file):

    pd.options.display.max_rows = 10
    pd.options.display.max_columns = 999
    black = pd.read_csv(file)
    black = pd.DataFrame(black)
    # Matrix
    ms.matrix(black)
    plt.show()
    # Bar plot
    ms.bar(black)
    plt.show()
    plt.show()
    # missing value
    black.isnull().any()
    # and applying son the entire data-set
    black.isnull().any().any()
    # number of missing null values in each column
    black.isnull().sum()

    class color:

        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    print(color.BOLD+"From visualization and isnull command I found out:\n")
    print("Product_Category_2 has 166986\t\t")
    print("Product_Category_3 has 3732299\n")
    black["Product_Category_2"] = black.Product_Category_2.fillna(black['Product_Category_2'].mean())
    black['Product_Category_3'] = black.Product_Category_3.fillna(black['Product_Category_3']).mean()
    print(black.head)
    print(color.BOLD + color.UNDERLINE+"replacing  the na values with mean and Removing inconsistency from the data set"
          + color.END+'\n')
    print("here I am changing all the lower case column names to uppercase\n")
    black.columns = black.columns.str.upper()
    print(black.columns)

    b = black.iloc[1:10]
    return b

# Data Visualization


def visualization(file):

    black = pd.read_csv(file)
    black = pd.DataFrame(black)
    black["Product_Category_2"] = black.Product_Category_2.fillna(black['Product_Category_2'].mean())
    black['Product_Category_3'] = black.Product_Category_3.fillna(black['Product_Category_3']).mean()
    black.columns = black.columns.str.upper()

    # counts = black["GENDER"].value_counts()
    # print('\nTotal number according to gender that is Male & Female\n', counts)

    # count_age = black["AGE"].value_counts()
    # type(count_age)  # checking the type of a series

    plt.subplots(figsize=(12, 7))
    sns.barplot(x='AGE', y='PURCHASE',hue='GENDER', data=black)
    # current_palette = sns.color_palette("Greens")

    sns.catplot(x="AGE", y='PURCHASE',hue='GENDER',kind='box',col="CITY_CATEGORY", data=black)

    plt.subplots(figsize=(12, 7))
    sns.countplot("CITY_CATEGORY", hue="GENDER", data=black)

    sns.catplot("GENDER", col="CITY_CATEGORY", col_wrap=3, data=black, kind="count")

    plt.subplots(figsize=(12, 7))
    sns.countplot(black['CITY_CATEGORY'], hue=black['AGE'])

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.pie(black['AGE'].value_counts(), explode=(0.1, 0, 0, 0, 0, 0, 0), labels=black['AGE'].unique(),
            autopct='%1.1f%%',
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.legend()

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    labels = ['0-17', '18-25', '26-35', '36-45', '46-50','51-55','50']
    ax1.pie(black.groupby('AGE')['PURCHASE'].sum(), labels=labels,
            explode=(0, 0, 0.1, 0.1, 0, 0, 0),
            autopct='%1.1f%%'
            , shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.legend()

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    labels = ['First Year', 'Second Year', 'Third Year', 'More Than Four Years', 'Geust']
    ax1.pie(black.groupby('STAY_IN_CURRENT_CITY_YEARS')['PURCHASE'].sum(), labels=labels, explode=(0.1, 0.1, 0, 0, 0),
            autopct='%1.1f%%'
            , shadow=True, startangle= 90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.legend()

    plt.subplots(figsize=(12, 7))
    sns.countplot(black['STAY_IN_CURRENT_CITY_YEARS'], hue=black['STAY_IN_CURRENT_CITY_YEARS'])

    return 0

# descriptive analysis


def descr_analysis(file):
    black = pd.read_csv(file)
    black = pd.DataFrame(black)
    black["Product_Category_2"] = black.Product_Category_2.fillna(black['Product_Category_2'].mean())
    black['Product_Category_3'] = black.Product_Category_3.fillna(black['Product_Category_3']).mean()
    black.columns = black.columns.str.upper()

    # Each attribute data type

    print(black.dtypes)

    # Descriptive Statistics
    print("Before correlation and numerical conversion is applied \n{}".format(black.describe(include='all')))

    print("CLASS DISTRIBUTION\n")
    print("According to: AGE\n")
    print(black.groupby('AGE').size())

    print("\nMarried and unmarried\n")
    print("0-unmarried :: 1-married\n")
    print(black.groupby('MARITAL_STATUS').size())

    print("\nAccording to city:\n")
    print(black.groupby('CITY_CATEGORY').size())

    print("\nAccording to the stay in each city:\n")
    print(black.groupby('STAY_IN_CURRENT_CITY_YEARS').size())

    print("\nAccording to Gender:\n")
    print(black.groupby('GENDER').size())

    # display(black.head())
    # black.drop(['USER_ID','PRODUCT_ID'],axis='columns')
    # black.convert_objects(convert_numeric= True)

    def handle_categorical_data(df):
        columns = df.columns.values

        for column in columns:
            text_dictionary_val = {}

            def convert_int(val):
                return text_dictionary_val[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_dictionary_val:
                        text_dictionary_val[unique] = x
                        x += 1
                df[column] = list(map(convert_int, df[column]))

        return df

    df = handle_categorical_data(black)
    # df
    # Correlation
    correlation = df.corr(method="pearson")
    # correlation
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()
    print("correlation chart:\n{}".format(correlation))
    return df.loc[1:10]

# Predictive analytics

def predictions(file):
    # file = 'BlackFriday.csv'
    black = pd.read_csv(file)
    black = pd.DataFrame(black)
    black["Product_Category_2"] = black.Product_Category_2.fillna(black['Product_Category_2'].mean())
    black['Product_Category_3'] = black.Product_Category_3.fillna(black['Product_Category_3']).mean()
    black.columns = black.columns.str.upper()

    black.head()
    # black["GENDER"]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df = black
    # black.columns
    df.PRODUCT_ID = le.fit_transform(df.PRODUCT_ID)
    # df["PRODUCT_ID"]

    # Dividing the train and test data set in 70 t0 30 percentage ratio
    train_data = df.iloc[:376303, :]

    test_data = df.iloc[376303:, :]

    def dummies_train_test(file_name):

        print("original features:\n", list(file_name.columns), '\n')
        d_dummies = pd.get_dummies(file_name)
        print("\nAfter applying dummies methodology\n", list(d_dummies.columns))

        x_train = d_dummies.drop('PURCHASE', axis=1)
        # X = d_dummies

        list(x_train.columns)
        y_train = d_dummies[['PURCHASE']]
        list(y_train.columns)

        return x_train, y_train

    x_train,y_train = dummies_train_test(train_data)
    x_test,y_test = dummies_train_test(test_data)

    model = LinearRegression()
    # help(model.fit)

    fit_linear = model.fit(x_train, y_train)
    print("Predictions for purchase\n", model.predict(x_test))
    print("\nScore for purchase\n", model.score(x_test, y_test))

    # Used in Logistic regression where one hot algorithm methodology is used
    # one hot encoding / also called dummy variable

    final_set = black.drop(['USER_ID', 'PRODUCT_ID'], axis='columns')
    type(final_set)

    print("original features:\n", list(final_set.columns), '\n')
    d_dummies = pd.get_dummies(final_set)
    print("Features after One-Hot Encoding:\n", list(d_dummies.columns))
    # d_dummies.columns.values

    features = d_dummies.iloc[:, [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values
    # features.columns.values
    X = features

    y = d_dummies['GENDER_F'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    print("Logistic regression score on the test set: {:.2f}".format(logreg.score(X_test, y_test)))
    prediction = logreg.predict(X_test)

    # predictions

    cls_report = classification_report(y_test, prediction)

    acc_score = accuracy_score(y_test, prediction)

    return cls_report, acc_score


    if __name__ == '__main__':
        extraction('../input/BlackFriday.csv')
        clean('../input/BlackFriday.csv')
        visualization('../input/BlackFriday.csv')
        descr_analysis("../input/BlackFriday.csv")
        predictions("../input/BlackFriday.csv")
    

