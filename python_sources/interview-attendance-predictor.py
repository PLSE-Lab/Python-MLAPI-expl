import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

def replace_yes_no(data, columns):
    # we want all values to be lowercase, and without any extra whitespace
    # note that we also convert nan to 'na', since the mapping works only for strings
    # we assume any string not equal to 'yes' or 'no' are equivalent to 'no'
    mapping = {'yes': 1, 'no' : 0}
    unique = set()
    for column in columns:
        data[column] = lower(data[column])
        unique.update(pd.unique(data[column]))
    for u in unique:
        if u != 'yes' and u != 'no':
            mapping[u] = 0
    return data[columns].replace(mapping)
    
def replace(data, column):
    # we want all values to be lowercase, and without any extra whitespace
    # note that we also convert nan to 'na', since the mapping works only for strings
    mapping = {}
    data[column] = lower(data[column])
    i = 0
    for u in pd.unique(data[column]):
        if mapping.get(u) == None:
            mapping[u] = i
            i += 1
    return data[column].replace(mapping)
    
def lower(column):
    return column.replace( {float('nan') : 'na'} ).map( lambda x : x.lower().strip() )
    
def create_mapping(arr, mapping = None):
    if mapping == None:
        mapping = {}
    i = 0
    for element in arr:
        if mapping.get(element) == None:
            mapping[element] = i
            i += 1
    return mapping

def parse_date(date):
    parsed_date = []
    temp = ''
    for letter in str(date):
        if letter.isdigit():
            temp += letter
        else:
            parsed_date.append(temp)
            temp = ''
    if temp != '':
        parsed_date.append(temp)
    return parsed_date

# get the data
data = pd.read_csv('../input/Interview.csv')
model = XGBRegressor(n_estimators=10000000000,learning_rate=0.01)
target = 'Observed Attendance'
predictors = []

# first, we want to convert the values in the target column to integers
data[target] = lower(data[target])
data[target] = data[target].replace( create_mapping(pd.unique(data[target]), {'yes' : 1, 'no' : 0}) )

# build y
y = data[target]

# now we build X
X = pd.DataFrame()

# these columns contain yes/no questions that were answered by the candidates
yes_no_columns = [
    'Have you obtained the necessary permission to start at the required time',
    'Hope there will be no unscheduled meetings',
    'Can I Call you three hours before the interview and follow up on your attendance for the interview',
    'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
    'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
    'Are you clear with the venue details and the landmark.',
    'Has the call letter been shared', 
    'Expected Attendance'
]
X[yes_no_columns] = replace_yes_no(data, yes_no_columns)

# next, we want to consider the dates of the interviews, in particular we care about the day and month
# we will create 2 new columns for them
days = []
months = []
for date in data['Date of Interview']:
    parsed_date = parse_date(date)
    if parsed_date[0] != '':
        days.append(int(parsed_date[0]))
    else:
        # if the interview date is not set, set as NaN for now
        days.append(float('nan'))
    if parsed_date[1] != '':
        months.append(int(parsed_date[1]))
    else:
        months.append(float('nan'))
# create the 2 columns
X['Day'] = Imputer().fit_transform( pd.DataFrame(days, dtype = np.int) )
X['Month'] = Imputer().fit_transform( pd.DataFrame(months, dtype = np.int) )

# we want to consider other columns as well
#X['Gender'] = replace(data, 'Gender')
#X['Client name'] = replace(data, 'Client name')
#X['Interview Type'] = replace(data, 'Interview Type')
X['Marital Status'] = replace(data, 'Marital Status')
#X['Industry'] = replace(data, 'Industry')

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)

predictions = model.predict(test_X)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

num_correct = 0
for actual,predicted in zip(test_y,predictions):
    if actual == predicted:
        num_correct += 1
mae = mean_absolute_error(test_y, predictions)
score = str(num_correct) +'/' + str(predictions.size)
print(mae)
print(score)

# save predictions
pd.DataFrame({'Predicted Attendance': predictions, 'Observed Attendance' : test_y }).to_csv('predictions.csv', index = False)
pd.DataFrame({'Mean Absolute Error': mae, 'Score' : score }, index=[0]).to_csv('summary.csv', index = False)