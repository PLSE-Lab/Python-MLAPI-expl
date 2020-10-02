# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy
exec(open("../input/evaluation.py").read())

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read competition data files:
#train = pd.read_csv("../input/training.csv")
#test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv("../input/check_agreement.csv")
check_correlation = pd.read_csv("../input/check_correlation.csv")

agreement_probs = numpy.random.uniform(0,1,len(check_agreement))
ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric on random generated probs', ks, ks < 0.09)

agreement_probs = numpy.arange(1,(1+len(check_agreement)))/len(check_agreement)

ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric on linearly increasing probs', ks, ks < 0.09)

correlation_probs = numpy.random.uniform(0,1,len(check_correlation))
cvm = compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

# Write summaries of the train and test sets to the log
#print('\nSummary of train dataset:\n')
#print(train.describe())
#print('\nSummary of test dataset:\n')
#print(test.describe())
