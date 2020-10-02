
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.impute import SimpleImputer

# Path of the file to read
my_imputer = SimpleImputer()
data = pd.read_csv("../input/whicsv/world-happiness-report-20192.csv")
print(data)
# Create target object and call it y
y = data.Positive_affect
print(y)
# Create X
features = ['Ladder','SD_of_Ladder','Negative_affect','Social_support','Freedom','Corruption','Generosity','Log_of_GDP_per_capita','Healthy_life_expectancy']
X = data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = linear_model.Ridge(alpha=.5)
# Fit Model
model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

output = pd.DataFrame({'Country_(region)': val_X.index,'Positive_affect': val_predictions})
output.to_csv('submission.csv', index=False)

print(val_predictions)