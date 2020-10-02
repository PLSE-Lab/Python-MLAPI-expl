import argparse
import itertools
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

DATA_FILE = '../input/HR_comma_sep.csv'

def read_data():
    return pandas.read_csv(DATA_FILE)

def plot_hours_satisfaction(data, color_label_column):
    fig = plt.figure()
    if data[color_label_column].dtype == float:
        colors = plt.cm.YlOrRd
        ax = fig.add_subplot(111)
        sc = ax.scatter(data.average_montly_hours, data.satisfaction_level, cmap=colors, c=data[color_label_column])
        fig.colorbar(sc)
    elif data[color_label_column].dtype == int:
        column_data = data[color_label_column]
        max_val = column_data.max()
        norm = mcolors.Normalize(vmin=0, vmax=max_val, clip=False)
        colors = plt.cm.YlOrRd
        ax = fig.add_subplot(111)
        sc = ax.scatter(data.average_montly_hours, data.satisfaction_level, cmap=colors, norm=norm, c=data[color_label_column])
        fig.colorbar(sc)
    elif data[color_label_column].dtype == 'object':
        column_values = data[color_label_column].unique()
        grouped = data.groupby(color_label_column)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(grouped)))
        colors_by_value = zip(column_values, colors)
        for val, color in colors_by_value:
            ax = fig.add_subplot(111)
            label_column_data = grouped.get_group(val)
            ax.scatter(label_column_data.average_montly_hours, label_column_data.satisfaction_level, color=color, label=val)
        plt.legend(loc='right', bbox_to_anchor=(1.35, 0.5))

    plt.title('hours vs satisfaction colored by %s' % color_label_column)
    plt.xlabel('average_monthly_hours')
    plt.ylabel('satisfaction_level')
    fig.savefig('hours_satisfaction_colored_by_%s.png' % color_label_column, bbox_inches='tight')

def plot_pairs_colored_by_left(data, columnx, columny):
    fig = plt.figure()
    colors = plt.cm.coolwarm
    columnx_data = data[columnx]
    if columnx_data.dtype == 'object':
        colx_vals = columnx_data.unique()
        columnx_data = columnx_data.apply(lambda entry: np.where(colx_vals==entry)[0][0])
    
    columny_data = data[columny]
    if columny_data.dtype == 'object':
        coly_vals = columny_data.unique()
        columny_data = columny_data.apply(lambda entry: np.where(coly_vals==entry)[0][0])
    
    ax = fig.add_subplot(111)
    sc = ax.scatter(columnx_data, columny_data, cmap=colors, c=data['left'], s=10)
    fig.colorbar(sc)

    plt.title('%s vs %s colored by left' % (columnx, columny))
    plt.xlabel(columnx)
    plt.ylabel(columny)
    fig.savefig('%s_vs_%s_colored_by_left.png' % (columnx, columny))

def do_decision_trees(data):
    # need to clean up some data
    str_columns = [col for col in data if data[col].dtype == 'object']
    le_map = dict()
    for col in str_columns:
        le = preprocessing.LabelEncoder()
        le.fit(data[col])
        le_map[col] = le
        data[col] = le.transform(data[col])
    clf = tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(data.drop('left', axis=1), data['left'], test_size=0.2, random_state=33)
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
def main():
    parser = argparse.ArgumentParser(description='do some machine learning '
                                     + 'on the data found at https://www.kaggle.com/ludobenistant/hr-analytics')
    parser.add_argument('--hoursvsat', dest='hoursvsat', action='store_true')
    parser.add_argument('--pairs', dest='pairs', action='store_true')
    parser.add_argument('--learn', dest='learn', action='store_true')
    args = parser.parse_args()
    
    data = read_data()

    print('plotting hours worked vs. satisfaction, colored by different columns')
    for column in data:
        print(column)
        plot_hours_satisfaction(data, column)
    print('plotting pairwise graphs, colored by whether the employee left or not')
    for (col_a, col_b) in itertools.product(data.columns, repeat=2):
        print('%s vs %s' % (col_a, col_b))
        plot_pairs_colored_by_left(data, col_a, col_b)
    print('apply decision trees algorithm')
    do_decision_trees(data)
        
if __name__ == "__main__":
    main()
