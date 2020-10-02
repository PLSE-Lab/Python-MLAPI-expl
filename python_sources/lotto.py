import numpy as np
import pandas as pd

predict_number = set()
columns_list = ["first","second","third","fourth","fifth","sixth"]

while len(predict_number) < 6:
    random_number = np.random.randint(1,46,[8145060,6])
    # 8145060 = (45*44*43*42*41*40) / (6*5*4*3*2*1)
    df = pd.DataFrame(random_number, columns=columns_list)
    
    freq_number_dict = {n:0 for n in range(1,46)}
    freq_number = pd.Series(freq_number_dict)
    
    for column in columns_list:
        freq_number = freq_number.add(df[column].value_counts(), fill_value=0)
    
    freq_number = freq_number.astype("int32").sort_values(ascending=False)
    most_freq_number = freq_number.index[0]
    predict_number.add(most_freq_number)
print(predict_number)


# ---------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import csv as csv
# from sklearn.ensemble import RandomForestClassifier

# # Data cleanup
# # TRAIN DATA
# train_df = pd.read_csv('../input/train.csv', header=0)        # Load the train file into a dataframe

# # I need to convert all strings to integer classifiers.
# # I need to fill in the missing values of the data and make it complete.

# # female = 0, Male = 1
# train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# # Embarked from 'C', 'Q', 'S'
# # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# # All missing Embarked -> just make them embark from most common place
# if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
#     train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

# Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
# Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
# train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# # All the ages with no data -> make the median of all Ages
# median_age = train_df['Age'].dropna().median()
# if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
#     train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
# train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# # TEST DATA
# test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe

# # I need to do the same with the test data now, so that the columns are the same as the training data
# # I need to convert all strings to integer classifiers:
# # female = 0, Male = 1
# test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)