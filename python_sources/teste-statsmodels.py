import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as smf

# Create the Breast Cancer dataframe
def breast_cancer_df():
    cancer = load_breast_cancer()
    data = np.c_[cancer.data, cancer.target]
    columns = np.append([x.replace(' ', '_') for x in cancer.feature_names], ["target"])
    return pd.DataFrame(data, columns=columns)

df = breast_cancer_df()
df.sample(5)


lr_model = smf.logit("target ~ mean_radius + mean_texture", df).fit()
print(lr_model.summary2())
