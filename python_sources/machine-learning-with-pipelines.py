#!/usr/bin/env python
# coding: utf-8

# This is my work through of the continually housing market competition on Kaggle.  This represents a second draft, with the addtion of piplines. Automation of encoding/imputing and parameter optimisation to come later

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("../input/train.csv")
Y = data.SalePrice
X = data.drop(["SalePrice"], axis=1)

test = pd.read_csv("../input/test.csv")

X.describe()


#  First things first, let's deal with NAs and clean the data. For simplicity, we will use numeric data only in this kernel.

# In[ ]:


#MSSubClass is actually categorical, not numerical, but NaN cannot be deduced from logic
X.MSSubClass = X.MSSubClass.astype("O")
test.MSSubClass = test.MSSubClass.astype("O")
#ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure
#BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual
#Functional, FirePlaceQu, GarageFinish, GarageQual 
#GarageCond, PavedDrive, PoolQC are object but has numerical relation (ex > gd)
X.replace({"ExterQual": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "ExterCond": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "BsmtQual": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
           "BsmtCond": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
           "BsmtExposure": {"Gd": 4,
                            "Av": 3,
                            "Mn": 2,
                            "No": 1,
                            np.nan: 0
                            },
           "BsmtFinType1": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "BsmtFinType2": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "HeatingQC": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "KitchenQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                        },
           "Functional": {"Typ": 7,
                          "Min1": 6,
                          "Min2": 5,
                          "Mod": 4,
                          "Maj1": 3,
                          "Maj2": 2,
                          "Sev": 1,
                          "Sal": 0
                          },
           "FireplaceQu": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageFinish": {"Fin": 3,
                            "RFn": 2,
                            "Unf": 1,
                            np.nan: 0
                          },
           "GarageQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "PavedDrive": {"Y": 3,
                            "P": 2,
                            "N": 1,
                            np.nan: 0
                          },
           "PoolQC": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          }
          }, inplace=True)
test.replace({"ExterQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1
                           },
              "ExterCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1
                           },
              "ExterCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                           },
           "BsmtQual": {"Ex": 5,
                        "Gd": 4,
                        "TA": 3,
                        "Fa": 2,
                        "Po": 1,
                        np.nan: 0
                        },
              "BsmtCond": {"Ex": 5,
                           "Gd": 4,
                           "TA": 3,
                           "Fa": 2,
                           "Po": 1,
                           np.nan: 0
                          },
           "BsmtExposure": {"Gd": 4,
                            "Av": 3,
                            "Mn": 2,
                            "No": 1,
                            np.nan: 0
                            },
           "BsmtFinType1": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "BsmtFinType2": {"GLQ": 6,
                            "ALG": 5,
                            "BLQ": 4,
                            "Rec": 3,
                            "LwQ": 2,
                            "Unf": 1,
                            np.nan: 0
                            },
           "HeatingQC": {"Ex": 5,
                         "Gd": 4,
                         "TA": 3,
                         "Fa": 2,
                         "Po": 1
                        },
           "KitchenQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                        },
           "Functional": {"Typ": 7,
                          "Min1": 6,
                          "Min2": 5,
                          "Mod": 4,
                          "Maj1": 3,
                          "Maj2": 2,
                          "Sev": 1,
                          "Sal": 0
                          },
           "FireplaceQu": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageFinish": {"Fin": 3,
                            "RFn": 2,
                            "Unf": 1,
                            np.nan: 0
                          },
           "GarageQual": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "GarageCond": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          },
           "PavedDrive": {"Y": 3,
                            "P": 2,
                            "N": 1,
                            np.nan: 0
                          },
           "PoolQC": {"Ex": 5,
                            "Gd": 4,
                            "TA": 3,
                            "Fa": 2,
                            "Po": 1,
                            np.nan: 0
                          }
          }, inplace=True)


# Essentially, I just labelled encoded a large chunk of the data. I elected to encode this chunk manually for two reasons. First, label encoding isn't always appropriate - because it introduces a numeric ranking, this can cause the model to detect orders where there are none, and I didn't want to introduce that error for columns that were unordered objects. And second, because these data could be expressed numerically, I wanted to include them to prevent data leakage.  Now I'll drop objects and double check NaNs...

# In[ ]:


XClean = X.select_dtypes(exclude=["O"])
#XClean.isna().sum() to see where NaN are
XClean.fillna(0, inplace=True)
testClean = test.select_dtypes(exclude=["O"])
#testClean.isna().sum() to see where NaN are
testClean.fillna(0, inplace=True)


# Since all the data in the numeric types can be simply inferred from the data description, that means both imputation and encoding aren't necessary. So prepare for the world's most basic piepline...

# In[ ]:


pipeline = Pipeline([#imputer 
                     #encoder
                     #feature select or reduction
                     ("model", XGBRegressor(random_state=0))
                    ])


# I'm now going to set up a `GridSearchCV` to find the optimal parameters. Much thanks to Aashita and here fantastic "Advanced Pipelines Tutorial" kernel! https://www.kaggle.com/aashita/advanced-pipelines-tutorial
# I'm electing not to use an early stopping rounds command for XGBoost as the grid search will ensure that the optimal result is picked no matter what. I'm still going to use a train test split to check for overfitting.

# In[ ]:


trainX, testX, trainY, testY = train_test_split(XClean, Y)
paramGrid = {
    "model__n_estimators": [10, 50, 100, 250, 500, 750, 1000],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
}

modelCV = GridSearchCV(pipeline, 
                       cv=5, 
                       param_grid=paramGrid)
modelCV.fit(trainX, trainY)


# Let's see what the parameters were...

# In[ ]:


modelCV.best_params_


# Ok now let's fit those to the model and test on the split set...

# In[ ]:


modelCV.refit
print(mean_absolute_error(testY, modelCV.predict(testX)))


# Seems reasonable, though it probably could be improved. It'll do for now. Lets retrain using the defined parameters on the whole data set, then write our results file.

# In[ ]:


pipeline = Pipeline([#imputer 
                     #encoder
                     #feature select or reduction
                     ("model", XGBRegressor(random_state=0,
                                            learning_rate=0.1,
                                            n_estimators=750
                                           ))
                    ])
pipeline.fit(XClean, Y)
output = pd.DataFrame({"Id": testClean.Id,
                       "SalePrice": pipeline.predict(testClean)
                      }
                     ).set_index("Id")
output.to_csv("submission.csv")


# In[ ]:




