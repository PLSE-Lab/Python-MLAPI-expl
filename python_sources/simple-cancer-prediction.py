import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print("Start of the Script\n")

# Load Data
data = load_breast_cancer()

# Print Data
df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target)
df = pd.concat([df_x, df_y], axis=1)
print(df.head(3))

# Create Model
model = LogisticRegression()

learningData_num = round(len(data.data)*0.8)

cancer_x = data.data[:, :10]
cancer_y = data.target
model.fit(cancer_x[0:learningData_num, :], cancer_y[0:learningData_num])

# Predict
y_pred = model.predict(cancer_x[learningData_num:, :])

# Evaluate Model
ac_score = accuracy_score(cancer_y[learningData_num:], y_pred)
print("Accuracy: " + str(ac_score) + "\n")

print("End of the Script")