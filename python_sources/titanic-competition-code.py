# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Load the dataset
train_file = '../input/train.csv'
test_file = '../input/test.csv'
submission_file = 'submission.csv'


def PrepareTrainData(in_file):
    full_data = pd.read_csv(in_file)
    
    # Print the first few entries of the RMS Titanic data
    display(full_data.head())
    
    # Store the 'Survived' feature in a new variable and remove it from the dataset
    outcomes = full_data['Survived']
    data = full_data.drop('Survived', axis = 1)
    return data, outcomes

data, outcomes = PrepareTrainData(train_file)

# Show the new dataset with 'Survived' removed
display(data.head())

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print(accuracy_score(outcomes[:5], predictions))

def predictions_1(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Pclass'] == 3:
            predictions.append(0)
        elif passenger['Sex'] == 'female':
            predictions.append(1)
        elif passenger['Age'] <= 10.0:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print(accuracy_score(outcomes, predictions))

# Now run on test data
data = pd.read_csv(test_file)
predictions = predictions_1(data)
submission = pd.DataFrame(data['PassengerId'])
submission['Survived'] = predictions
print(submission)

submission.to_csv(submission_file, index=False)