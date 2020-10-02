#**************************************************************************
# Acknowledgement: File use: KaggleWord2VecUtility for token creation from:
# Project: Kaggle tutorial "Deep learning goes to the movies"  
# KaggleWord2VecUtility uses NTLK methods
# To Do: 
# 1. Bag of Words vs Word Embeddings
# 2. TFIDF
# 3. Cross validation
# 4. Experiment with different n_estimators
# *************************************************************************

from collections import defaultdict 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

from sklearn import cross_validation #, neighbors
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import confusion_matrix, roc_curve, auc

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

clean_train_reviews = []
train_data_features = []
ROC = defaultdict(list)

for i in range( 0, len(train["text"])):
   #words is a list of words - for each tweet
   words = KaggleWord2VecUtility.review_to_wordlist(train["text"][i], True)  
   clean_train_reviews.append(" ".join(words))

# ****** After removing stop words, create a bag of words from the training set *************
print ("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
np.asarray(train_data_features)
forest = RandomForestClassifier(n_estimators = 100) 

#Generator or numpy 

for hold_out_percent in np.arange (0.2,0.9,0.1):     
       
       
       dict_key = float(format(hold_out_percent, '.1f')) 
       
      
       X_train, X_test, Y_train, Y_test = \
       cross_validation.train_test_split(train_data_features, train["target"],\
                                         test_size= dict_key)                                               
       #print ("Training the random forest (this may take a while)...")       
       forest_train = forest.fit(X_train, Y_train)       
       
      
       predicted_Y_test_hold_out = forest_train.predict(X_test)  
       
       #Another approach I tried but does not add much value
       #Compare predicted_Y_test_hold_out and Y_test
       #tn, fp, fn, tp = confusion_matrix(predicted_Y_test_hold_out, \
       #                      Y_test).ravel()                
       #lst_vals.extend([tn, fp, fn, tp])
             
       false_positive_rate, true_positive_rate, thresholds = \
                                  roc_curve(Y_test, predicted_Y_test_hold_out)
                                  
       roc_auc = auc(false_positive_rate, true_positive_rate)
            
       ROC[dict_key] = format(roc_auc, '.2f')
       print ("Area Under Curve for hold_out of: " + str(dict_key) + " is " + str(roc_auc))
             
# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print ("Cleaning and parsing the tweets...\n")
for i in range(0,len(test["text"])):
      #words is a list of words - for each tweet
      words = KaggleWord2VecUtility.review_to_wordlist(test["text"][i], True)  
      clean_test_reviews.append(" ".join(words))
   
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)

#Which hold_out_set had the highest AUC. Use that model on the test set
#If auc_area is same for more then one holdout, just select first index 
max_val = max(ROC.values())
max_auc_hold_out_val = hold_out_set_for_test =\
                        [ k for (k,v) in ROC.items() if v >= max_val][0]
                        
print ("Fit the random forest with: " +  str(max_auc_hold_out_val))
                      
X_train, X_test, Y_train, Y_test = \
cross_validation.train_test_split(train_data_features, train["target"],\
                                         test_size= max_auc_hold_out_val)      
forest_test = forest.fit(X_train, Y_train)     
       
#Use the random forest to make sentiment label predictions
print ("Predicting test labels...\n")
result = forest_test.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "target" column
output = pd.DataFrame( data={"id":test["id"], "target":result} )

# Use pandas to write the comma-separated output file
output.to_csv(("./data/Bag_of_Words_model.csv"), index=False, quoting=3)
print ("Wrote results to Bag_of_Words_model.csv")
