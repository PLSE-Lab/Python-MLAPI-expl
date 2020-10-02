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
"""
#Master list of anything possible:  https://github.com/bulutyazilim/awesome-datascience
#Where to start-quora answer(Rohit):https://www.quora.com/How-does-a-total-beginner-start-to-learn-machine-learning-if-they-have-some-knowledge-of-programming-languages#
#Articles on ML:                    https://medium.com/towards-data-science/machine-learning/home
#CheatSheets collection:            https://unsupervisedmethods.com/cheat-sheet-of-machine-learning-and-python-and-math-cheat-sheets-a4afe4e791b6
#Statistical forecasting, notes on regression and time series analysis: https://www.thespreadsheetguru.com/blog/2014/7/7/5-different-ways-to-find-the-last-row-or-last-column-using-vba
#P/E Ratio for an Index:            http://www.financialwisdomforum.org/gummy-stuff/PE-ratios.htm
#Very useful chart on statistics:   https://static.coggle.it/diagram/WMesPqSf-AAB62b8
#ML model choosing chart:           http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#good website for basic python help:https://chrisalbon.com/#Python
#Log Reg/SVM/Dec Tree explained:    https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part1/
                                    https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part2/
#Algorithm for all models in python:https://spark.apache.org/docs/2.1.1/ml-classification-regression.html
#Models explained:                  https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/
#Different ways to create DataFrame:http://pbpython.com/pandas-list-dict.html
#Ways of scaling in python:         http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
#Complete tutorial on concat/merge: https://pandas.pydata.org/pandas-docs/stable/merging.html
#XGBoost explained:                 https://www.kaggle.com/humananalog/xgboost-lasso
#Gradient boost (XGB) explained:    http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
#Basics of EDA explained:           http://www.stat.cmu.edu/~hseltman/309/Book/chapter4.pdf
#Links to useful resources in python:https://unsupervisedmethods.com/my-curated-list-of-ai-and-machine-learning-resources-from-around-the-web-9a97823b8524
#PercentileOfscore:                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html#scipy.stats.percentileofscore
#ScoreofPercentile:                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.scoreatpercentile.html#scipy.stats.scoreatpercentile
#Stack/Unstack explained:           http://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/
#converting groupby to a dataframe: https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe
#Working with pipelines:            https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/
"""