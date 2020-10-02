#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train_transformed = pd.read_csv("../input/avito-data-translation-and-transformation/train_transformed.csv")
test_transformed = pd.read_csv("../input/avito-data-translation-and-transformation/test_transformed.csv")
test_transformed.head(5)
# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X,y = train_transformed.iloc[:, [3,4,5,6,7,8,9,10,11,12,14,17]], train_transformed.iloc[:,[18]]
#X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.fillna(value=0)
X = X.astype(np.float32)
y = y.values.ravel()

regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=10, verbose=0, warm_start=False)

regr.fit(X, y)

print(regr.feature_importances_)
regr.score(X,y)


# In[ ]:


deal_probability = regr.predict(test_transformed.iloc[:, [3,4,5,6,7,8,9,10,11,12,14,17]].fillna(value=0))
deal_prob = [x if x>0 else 0 for x in deal_probability]

#print deal_prob
#submission_op = pd.DataFrame(user_id = test_transformed['user_id'], deal_probability=deal_probability)
submission_op = pd.DataFrame({'item_id': test_transformed['item_id'], 'deal_probability': deal_prob})
submission_op = submission_op[['item_id', 'deal_probability']]
submission_op.to_csv('submission.csv', index=False)

