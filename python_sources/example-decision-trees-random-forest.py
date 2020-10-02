import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree, ensemble


def plot_confusion_matrix(y_test, y_pred, title):
    ''' Given a test set and a predicted set, plot the confusion matrix '''
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        square=True,
        annot=True,
        cbar=False,
        xticklabels=['healthy', 'diseased'],
        yticklabels=['healthy', 'diseased'],
        cmap='Blues'
    )
    plt.title(title)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    ax.set_ylim(2, -0.5)
    plt.show()


def create_tree_file(model, feature_names, output_path):
    ''' Given a decision tree model and the name of the features used, create
        a graphviz file '''
    import graphviz
    dot_data = tree.export_graphviz(
        model,
        out_file=output_path,
        feature_names=feature_names,
        filled=True,
        rounded=True
    )


def decision_tree_results(features, classes, test_size, output_path):
    ''' Given a set of classes and a set of features, plot the confusion matrix
        and create a graphviz file '''
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=test_size, shuffle=True, random_state=0)

    classification_tree = tree.DecisionTreeClassifier()
    classification_tree = classification_tree.fit(X_train, y_train)
    print(f"\taccuracy: {classification_tree.score(X_test, y_test)}")
    y_pred = classification_tree.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, 'Confusion matrix of decision tree classifier on heart disease data')
    create_tree_file(classification_tree, X_train.columns.values, output_path)


def random_forest_results(features, classes, test_size, n_trees, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=test_size, shuffle=True, random_state=0)

    random_forest = ensemble.RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=0,
    )
    random_forest = random_forest.fit(X_train, y_train)
    print("random forest:")
    print(f"\tnumber of trees: {n_trees}")
    print(f"\tmaximum depth of each tree: {max_depth}")
    print(f"\taccuracy: {random_forest.score(X_test, y_test)}")
    y_pred = random_forest.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, 'Confusion matrix of random forest classifier on heart disease data (full features)')


# import dataset
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
y = df['target']

# train a decision tree model using 'most important' features
selected_features = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X_reduced = df[selected_features]
print("decision tree - selected features:")
decision_tree_results(X_reduced, y, 0.25, "tree_file_reduced")

# train a decision tree model using all features
X_full = df.drop('target', axis=1)
print("decision tree - all features:")
decision_tree_results(X_full, y, 0.25, "tree_file_full")

# train a random forest model using all features
random_forest_results(X_full, y, 0.25, 1000, 20)
