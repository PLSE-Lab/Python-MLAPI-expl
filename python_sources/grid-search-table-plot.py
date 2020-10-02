# ***********
# * WARNING *
# ***********
#
# I've evolved this utility script to other at:
#
# https://www.kaggle.com/juanmah/grid-search-utils
# 
# I'll delete this notebook in some days.
#
# Please, go to my new script or copy or fork this one

# Displaying the results of a Grid Search are very useful ways to parametrize an estimator.
# This notebook shows a function that displays the results in a helpful way.

import pandas as pd
from matplotlib import pyplot as plt

def grid_search_plot(grid_clf,
                     parameters):
    scores_df = pd.DataFrame(grid_clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])
    best_row = scores_df.iloc[0, :]
    best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    
    rows = -(-len(parameters) // 3)
    columns = min(len(parameters), 3)
    index = 0
    
    plt.figure(figsize=(columns * 5, rows * 5))
    plt.subplots_adjust(hspace=0.4)
    
    for param in parameters:
               
        best_param = best_row['param_' + param]
        scores_df = scores_df.sort_values(by='param_' + param)
        means = scores_df['mean_test_score']
        stds = scores_df['std_test_score']
        params = scores_df['param_' + param]

        index += 1
        plt.subplot(rows, columns, index)
        plt.plot(best_param, best_mean, 'or')

        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')

        plt.title(param + " vs Score\nBest Score {:0.5f}".format(grid_clf.best_score_))
        plt.xlabel(param)
        plt.ylabel('Score')

    plt.show()
    

def grid_search_table_plot(grid_clf,
                           param_name,
                           num_results=15,
                           negative=True,
                           graph=True,
                           display_all_params=True):
    """Display grid search results

    Parameters
    ----------

    grid_clf : estimator
        The estimator resulting from a grid search
        for example: grid_clf = GridSearchCV( ...

    param_name : string
        A string with the name of the parameter being tested

    num_results : integer, default: 15
        The number of results to display

    negative : boolean, default: True
        Should the sign of the score be reversed?
        Scoring = 'neg_log_loss', for instance

    graph : boolean, default: True
        Should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params : boolean, default: True
        Should we print out all of the parameters, not just the ones searched for?

    Examples
    --------
    iris = datasets.load_iris()

    grid_clf = GridSearchCV(estimator  = svm.SVC(), 
                            param_grid = {'C': range(1,10)}, 
                            cv         = 10)

    _ = grid_clf.fit(iris.data, iris.target)
    
    GridSearch_table_plot(grid_clf, "C", negative=False)
    """
    
    from matplotlib import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by=['rank_test_score', 'mean_fit_time'])

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()
