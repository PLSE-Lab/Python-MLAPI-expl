import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

#definisco una funizone per il calcolo di L2 distanza euclidea

def distanza (row1, row2, length):

    distance=0

    for x in range(length):

        distance+=pow(row1[x]-row2[x],2) #quadrato della differenza sommato posto per posto

    return math.sqrt(distance) #radice della somma dei quadrati delle differenze = distanza euclidea
