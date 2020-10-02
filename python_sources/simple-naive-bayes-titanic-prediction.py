import pandas as pd

# Load inputs
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# The survival odds for each (Sex, Pclass) combination are calculated based on its own ratio of survivors
survival_odds = df_train.groupby(['Sex', 'Pclass', 'Survived']).size().to_frame('PassengerAmount')
survival_odds['SurvivalOdds'] = survival_odds.groupby(level=[0, 1]).sum()
survival_odds['SurvivalOdds'] = survival_odds['PassengerAmount'] / survival_odds['SurvivalOdds']
survival_odds = survival_odds.drop(0, level=2, axis=0).reset_index()[['Sex', 'Pclass', 'SurvivalOdds']]

# Makes the prediction on the test data
# If the odds are greater than 50%, then the prediction is that this individual will survive
df_test = pd.merge(df_test, survival_odds, on=['Sex', 'Pclass'])
df_test['Survived'] = df_test['SurvivalOdds'].apply(lambda odd : 1 if odd >= 0.5 else 0)

# Create submission file
df_test[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)