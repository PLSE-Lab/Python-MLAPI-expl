import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# The competition datafiles are in the directory ../input
# Read competition data files:
train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

# Write to the log:

print("Partitioning training data into cross-validation and training...")

cross_validation_data = train_data.sample(frac = 0.2)
train_data_x_cv = train_data.drop(cross_validation_data.index)


print("Length of original training set "+str(len(train_data)))
print("Length of cross-validation set "+str(len(cross_validation_data)))
print("Length of final training set "+str(len(train_data_x_cv)))

x_variables = list(train_data.columns)
x_variables.remove('label')
y_variable = 'label'

temp_df = pd.DataFrame(train_data[x_variables].sum())
temp_df.columns = ['sum_of_intensities']
#temp_df.reset_index(inplace = True)
temp_df.sort_values(by = ['sum_of_intensities'],ascending = [1],inplace = True)

temp_df.reset_index(inplace = True)
temp_df.columns = ['features','sum_of_intensities']

features_to_remove = list(temp_df.loc[temp_df['sum_of_intensities']==0,'features'])
print("Following are the features that neeed to be removed \n")
print(features_to_remove)
x_variables_copy = x_variables[:]
print("\n These features will now be removed \n")
for f in features_to_remove:
    x_variables_copy.remove(f)
    

# Neural Netwroks

Y = train_data_x_cv[y_variable].as_matrix()
X = train_data_x_cv[x_variables_copy].as_matrix()

print("Training neural nets... ")
my_classifier = MLPClassifier(solver = 'adam',alpha = 0.5,hidden_layer_sizes = (200,100),activation = 'logistic')
my_classifier.fit(X,Y)
predicted = my_classifier.predict(cross_validation_data[x_variables_copy].as_matrix())

print("Getting Accuracy for CV...")
accu = accuracy_score(cross_validation_data[y_variable].as_matrix(),predicted)
print("Accuracy = "+str(accu))

predicted_test_labels = my_classifier.predict(test_data[x_variables_copy])
data_dict = {'ImageId':range(1,len(predicted_test_labels)+1),'Label':predicted_test_labels}
test_labels = pd.DataFrame(data_dict)
test_labels.to_csv(r'Output_neural_nets.csv',index = False)