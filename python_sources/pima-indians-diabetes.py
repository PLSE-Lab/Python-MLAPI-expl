# %% constants

_id = "ID"

# %% imports

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# %% data

train = pd.read_csv("../input/train.csv", index_col=_id)
test = pd.read_csv("../input/test.csv", index_col=_id)
validate = pd.read_csv("../input/validate.csv", index_col=_id)

# %% columns

class columns:
    pass

c = columns()

for col in train.columns.values:
    setattr(c, col, col)

setattr(c, "GTT", "GTT")
setattr(c, "MultiplePregnancies", "MultiplePregnancies")
setattr(c, "Hypertensive", "Hypertensive")
setattr(c, "Obese", "Obese")
setattr(c, "IR", "IR")
setattr(c, "FH", "FH")
setattr(c, "AR", "AR")

# %% any missing data?

train.isna().sum()
test.isna().sum()

# %% Glucose

"""
set GTT = 1 if positive test

Rationale:
ADA criteria for positive two-hour 75 gm oral glucose tolerance test for gestational diabetes diagnosis is >= 153 mg/dL
"""

train[c.GTT] = 0
train.GTT.where(train.Glucose < 153, 1, inplace=True)
train.drop([c.Glucose], axis=1, inplace=True)

test[c.GTT] = 0
test.GTT.where(test.Glucose < 153, 1, inplace=True)
test.drop([c.Glucose], axis=1, inplace=True)

validate[c.GTT] = 0
validate.GTT.where(validate.Glucose < 153, 1, inplace=True)
validate.drop([c.Glucose], axis=1, inplace=True)

# %% Pregnancies

"""
MultiplePregnancies = 1 if Pregnancies > 6

Rationale:
Multiple gestations is a risk factor.
Correlation coefficient increases after 6 pregnancies.
"""
corrcoef = []
for k in range(train.Pregnancies.max()):
    subset = train[train.Pregnancies > k]
    corr = subset.corr()
    corrcoef.append(corr.Outcome.Pregnancies)


corrcoef = pd.Series(corrcoef)
corrcoef.plot()

train[c.MultiplePregnancies] = 0
train.MultiplePregnancies.where(train.Pregnancies < 6, 1, inplace=True)
train.drop([c.Pregnancies], axis=1, inplace=True)

test[c.MultiplePregnancies] = 0
test.MultiplePregnancies.where(test.Pregnancies < 6, 1, inplace=True)
test.drop([c.Pregnancies], axis=1, inplace=True)

validate[c.MultiplePregnancies] = 0
validate.MultiplePregnancies.where(validate.Pregnancies < 6, 1, inplace=True)
validate.drop([c.Pregnancies], axis=1, inplace=True)

# %% BloodPressure

"""
Hypertensive = 1 if BloodPressure >=80

Rationale:
Hypertensive is defined if diastolic blood pressure >= 80
"""

train[c.Hypertensive] = 0
train.Hypertensive.where(train.BloodPressure < 80, 1, inplace=True)
train.drop([c.BloodPressure], axis=1, inplace=True)

test[c.Hypertensive] = 0
test.Hypertensive.where(test.BloodPressure < 80, 1, inplace=True)
test.drop([c.BloodPressure], axis=1, inplace=True)

validate[c.Hypertensive] = 0
validate.Hypertensive.where(validate.BloodPressure < 80, 1, inplace=True)
validate.drop([c.BloodPressure], axis=1, inplace=True)

# %% SkinThickness

"""
Drop observation

Rationale:
Skinfold thickness has little clinical predictive value.
"""

train.drop([c.SkinThickness], axis=1, inplace=True)
test.drop([c.SkinThickness], axis=1, inplace=True)
validate.drop([c.SkinThickness], axis=1, inplace=True)

# %% BMI

"""
Obese = 1 if BMI >= 30

Rationale:
Obesity of BMI >= 30 kg/m2 is known to increase type-2 diabetes risk by 50 to 75 percent.  (Interestingly, adipogensis is protective of diabetes for a term until Leptin resistence.  It's leptin resistence that leads to insulin resistence.)
"""

train[c.Obese] = 0
train.Obese.where(train.BMI < 30, 1, inplace = True)
train.drop([c.BMI], axis=1, inplace=True)

test[c.Obese] = 0
test.Obese.where(test.BMI < 30, 1, inplace = True)
test.drop([c.BMI], axis=1, inplace=True)

validate[c.Obese] = 0
validate.Obese.where(validate.BMI < 30, 1, inplace = True)
validate.drop([c.BMI], axis=1, inplace=True)

# %% insulin

"""
IR = 1 if Insulin > 166

Rationale:
16 - 166 mIU/L is the reference range for insulin 2 hours post glucose challenge and beyond is a sign of insulin resistence.
"""

train[c.IR] = 0
train.IR.where(train.Insulin < 166, 1, inplace = True)
train.drop([c.Insulin], axis=1, inplace=True)

test[c.IR] = 0
test.IR.where(test.Insulin < 166, 1, inplace = True)
test.drop([c.Insulin], axis=1, inplace=True)

validate[c.IR] = 0
validate.IR.where(validate.Insulin < 166, 1, inplace = True)
validate.drop([c.Insulin], axis=1, inplace=True)

# %% DiabetesPedigreeFunction

"""
FH = 1 if DiabetesPedigreeFunction >= 1

Rationale:
Family history of diabetes is a strong predictor or an individual developing the disease.
"""

train[c.FH] = 0
train.FH.where(train.DiabetesPedigreeFunction < 1, 1, inplace = True)
train.drop([c.DiabetesPedigreeFunction], axis=1, inplace=True)

test[c.FH] = 0
test.FH.where(test.DiabetesPedigreeFunction < 1, 1, inplace = True)
test.drop([c.DiabetesPedigreeFunction], axis=1, inplace=True)

validate[c.FH] = 0
validate.FH.where(validate.DiabetesPedigreeFunction < 1, 1, inplace = True)
validate.drop([c.DiabetesPedigreeFunction], axis=1, inplace=True)

# %% Age

"""
AR = 1 if Age > 25

Rationale:
Age > 25 is considered a risk factor for gestational diabetes.
"""

train[c.AR] = 0
train.AR.where(train.Age <= 25, 1, inplace=True)
train.drop([c.Age], axis=1, inplace=True)

test[c.AR] = 0
test.AR.where(test.Age <= 25, 1, inplace=True)
test.drop([c.Age], axis=1, inplace=True)

validate[c.AR] = 0
validate.AR.where(validate.Age <= 25, 1, inplace=True)
validate.drop([c.Age], axis=1, inplace=True)

# %% Build model

train_Y = train[[c.Outcome]]
train_X = train.drop([c.Outcome], axis=1)
test_Y = validate.Outcome

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_Y, verbose=False, early_stopping_rounds=5, eval_set=[(test, test_Y)])

# %% Test model

predictions = model.predict(test)
print()
print("Mean Absolute Error : " + str(mean_absolute_error(predictions.round(), test_Y)))
