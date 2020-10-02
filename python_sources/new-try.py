import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

#Print you can execute arbitrary python code
df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

df = df.drop(["Ticket", "Cabin"], axis=1)
df = df.dropna()


formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)"
results = {}

y,x = dmatrices(formula, data=df, return_type='dataframe')

model = sm.Logit(y,x)

# fit our model to the training data
res = model.fit()

# save the result for outputing predictions later
results['Logit'] = [res, formula]
res.summary()
