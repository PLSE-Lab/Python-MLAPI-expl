#Author: Tanya Ganesan
#Project: Iris-Dataset using Logistic Regression
#Version 1.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#### Returns sigmoid function as output.
#### The probability of the output being 0 - 1.
def sigmoid(x):
    return 1/(1+math.exp(-x))

#### Returns cost function for logistic regression
#### Input - X, theta, regularization parameter. Output - cost
#### the logarithm of 0 is set as a really high value

def costFunction(x_data_features_df,theta_np,y_df,lambda_):
    size = x_data_features_df.shape
    m = size[0]
    z_sr = x_data_features_df.dot(np.transpose(theta_np))
    h_sr = z_sr.apply(sigmoid)
    h_sr_1 = h_sr.apply(lambda x: (1.0 - x))
    log_h_sr = h_sr.apply(lambda x: -1000000000 if x==0 else math.log(x))
    y_1_df = y_df.apply(lambda x: (1.0 - x))
    log_h_sr_1 = h_sr_1.apply(lambda x: -1000000000 if x==0 else math.log(x))
    cost_term1_1 = -log_h_sr.dot(y_df)-log_h_sr_1.dot(y_1_df)
    cost_term1_2 = (1/m)*cost_term1_1
    cost_term2_1 = np.matmul(theta_np[1:],theta_np[1:])
    cost_term_2_2 = np.multiply(lambda_/(2*m),cost_term2_1)
    cost = cost_term1_2 + cost_term_2_2
    return cost.type

#### Gradient for logistic regression
#### Input - X, theta, Y, regularization parameter

def gradient(x_data_features_df,theta_np,y_df,lambda_):
    size = x_data_features_df.shape
    m = size[0]
    z_sr = x_data_features_df.dot(np.transpose(theta_np))
    h_sr = z_sr.apply(sigmoid)
    h = h_sr.to_frame()
    h_sr_y_df = h.subtract(y_df['type'],axis = 'index')
    term1_1 = np.matmul(np.transpose(h_sr_y_df.values),x_data_features_df.values)
    grad_term1 = np.multiply((1/m),term1_1)
    grad_term2 = np.multiply((lambda_/m),theta_np)
    grad = grad_term1 + grad_term2
    return grad

#### A method to optimize the value of theta.
#### Input - X, Y, theta, learning rate, number of iterations for convergence, regularization parameter
def gradient_descent(x_data_features_df, y_df, theta_np, alpha, num_iters,lambda_,species_name):
    size = y_df.shape
    m = size[0]
    theta_ = theta_np
    grad_ = theta_np
    costIter_list = []
    for i in range(num_iters):
        grad_ = gradient(x_data_features_df,theta_[0],y_df,lambda_)
        theta_ = theta_ + np.multiply(alpha,grad_)
        cost = costFunction(x_data_features_df,theta_[0],y_df,lambda_)
        costIter_list.append(cost)
    theta_optimised_np = theta_
    grad_optimised_np = grad_
    optimised_theta_grad_cost_list = [costIter_list,theta_optimised_np,grad_optimised_np]
    z_ = x_data_features_df.dot(np.transpose(theta_optimised_np))
    h_ = z_[0].apply(sigmoid)
    J_iter_plot = plt.figure()
    plt.plot(range(num_iters),costIter_list)
    plt.title("Learning curve for the training set" + " " + species_name ,fontsize = 12, fontstyle = 'normal' )
    plt.xlabel("No. of iterations",fontsize = 8, fontstyle = 'italic')
    plt.ylabel( " Cost J(" + r"$\theta$" + ")",fontsize = 8, fontstyle = 'italic' )
    plt.show()
    return optimised_theta_grad_cost_list

#### Split the set into training, validation, test set given the ratio in which it is to be split
def extract_sets(train_set,validation_set,test_set,data_species_type_df):
    train_index_type= int(data_species_type_df.shape[0]*train_set)
    validation_index_type = int(data_species_type_df.shape[0]*validation_set)
    test_index_type= int(data_species_type_df.shape[0]*test_set)
    y_type_train_df = data_species_type_df[0:train_index_type]
    y_type_validation_df = data_species_type_df[train_index_type:train_index_type+validation_index_type]
    y_type_test_df = data_species_type_df[train_index_type+validation_index_type:train_index_type+validation_index_type+test_index_type]
    extracted_sets_list = [y_type_train_df,y_type_validation_df,y_type_test_df]
    return(extracted_sets_list)

#### The output needs to be coverted to a numeric value
#### This is a problem of one to all. Hence three different datasets needs to be trained.
#### For the species to be pedicted the corresponding output needs to be 1 and the others 0 given the training set chosen
#### Input -  the set (could be training, validation or test),
#### So if the dataset passed is Iris_something, the 'type' column should be 1 for Iris_something and 0 for Iris_not_something
def numeric_conversion_set(set_df, species_string, species_types_np):
    x_data_features_df = set_df[['x0','x1','x2','x3','x4']]
    y_df = set_df[['type']]
    not_species1_list = []
    for i in species_types_np:
        if i != species_string:
            not_species1_list.append(i)
    y_species_df = y_df.replace(to_replace=[species_string,not_species1_list[0],not_species1_list[1]], value=[1.0,0.0,0.0])
    return(x_data_features_df, y_species_df,species_string)

#### Model the training set
#### theta is initially set to random values
def model_set_optimisation(x_data_features_df, y_species_df,species_name,lambda_,alpha,num_iters):
    feature_count = x_data_features_df.shape[1]
    theta_initial_np = np.random.randint(1,size=(1,feature_count))
    model_cost_theta_grad_list = gradient_descent(x_data_features_df, y_species_df, theta_initial_np, alpha, num_iters, lambda_,species_name)
    return(model_cost_theta_grad_list)

#### Once the best model is obtained we need to check how well it predicts
#### So after model paramters are obtained, they are put back to check how well it is performing
#### So if, h(x) > 0.5 and y = 1 and h(x) < 0.5 and y = 0, it is an indication that the model is predicting well
#### The functions compute_heuristic and training_error have been used to see how well the model performs

def compute_heuristic(set_df,theta_optimsed_np,model):
    x_data_features_df = set_df[model]
    z_sr = x_data_features_df.dot(np.transpose(theta_optimsed_np))
    h_sr = z_sr.apply(sigmoid)
    return(h_sr)

def training_error(set_df,theta_optimsed_np,model,y_df,title_name):
    training_error = []
    h_sr = compute_heuristic(set_df,theta_optimsed_np,model)
    y_prediction_sr = h_sr.apply(lambda x: 1.0 if x>=0.5 else 0.0)
    count_correct_positives = 0
    count_correct_negatives = 0
    count_wrong_negatives = 0
    count_wrong_positives = 0
    positives_size = y_df.groupby(['type']).get_group(1.0).shape[0]
    negatives_size = y_df.groupby(['type']).get_group(0.0).shape[0]
    y_df_list = list(y_df['type'])
    y_prediction_sr_list = list(y_prediction_sr)
    for i in range(y_df.shape[0]):
        if y_prediction_sr_list[i] == 1.0 and y_df_list[i]  == 1.0:
            count_correct_positives = count_correct_positives + 1
        if y_prediction_sr_list[i]  == 0.0 and y_df_list[i]  == 0.0:
            count_correct_negatives = count_correct_negatives + 1
        if y_prediction_sr_list[i]  == 1.0 and y_df_list[i]  == 0.0:
            count_wrong_positives = count_wrong_positives + 1
        if y_prediction_sr_list[i]  == 0.0 and y_df_list[i]  == 1.0:
            count_wrong_negatives = count_wrong_negatives + 1

    count_correct_positives_measure = (count_correct_positives/positives_size)*100.0
    training_error.append(count_correct_positives_measure)
    count_correct_negatives_measure = (count_correct_negatives/negatives_size)*100.0
    training_error.append(count_correct_negatives_measure)
    count_wrong_positives_measure = (count_wrong_positives/positives_size)*100.0
    training_error.append(count_wrong_positives_measure)
    count_wrong_negatives_measure = (count_wrong_negatives/negatives_size)*100.0
    training_error.append(count_wrong_negatives_measure)
    plt.suptitle(title_name,fontsize = 14)
    plt.subplot('211')
    plt.pie([training_error[0],training_error[2]],labels = ["Correct Positives","Wrong Positives"],autopct='%1.1f%%')
    plt.axis('equal')
    plt.subplot('212')
    plt.pie([training_error[1],training_error[3]],labels = ["Correct Negatives","Wrong Negatives"], autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    return(training_error)

##### Once the model parameters for all the three species are obtained, they are used to calculate h(x)
##### Of the three, whichever is maximum is predicted to the species it belongs to
def predict_species(set_df,theta_species_list,model_species_list):
    h_list = []
    predict_species = []
    for theta,model in zip(theta_species_list,model_species_list):
        set_model_df =  set_df[model]
        h_list.append(compute_heuristic(set_model_df,theta,model))
    size_h_list = h_list[0].shape[0]
    prob_species_y = []
    for i in range(size_h_list):
        temp_prob_species_y = []
        for j in range(len(h_list)):
            temp_prob_species_y.append(h_list[j][i])
        prob_species_y.append(temp_prob_species_y)
    for prob_y in prob_species_y:
        temp_prob_np = np.array([])
        temp_prob_np =  np.append(temp_prob_np, prob_y)
        index_max = indexMax(temp_prob_np)
        predict_species.append(index_max+1)
    return(predict_species)

#### A function which returns the index in array which has maximum value
def indexMax(list_np):
    max = 0.0
    for i in range(list_np.size):
        if list_np[i] > max:
            max = list_np[i]
            index = i
    return index

#### A function which returns the index in array which has minimum value
def indexMin(list_np):
    min = 100.0
    for i in range(list_np.size):
        if list_np[i] < min:
            min = list_np[i]
            index = i
    return index

##### Once, the species are predicted, the fucntion accuracy_prediction is used to check if it has indeed predicted right
def accuracy_prediction(species_actual_df_list, species_predicted_list):
    count = 0
    for i in range(len(species_actual_df_list)):
        if species_actual_df_list[i] == species_predicted_list[i]:
            count = count + 1
    m = len(species_actual_df_list)
    if m != 0:
        accuracy = (count/m )*100
    else:
        accuracy = "The set has no elements"
    return accuracy


#### Many factors influence the performance of the model - learning rate, iterations, regularization parameter, number of features, size of the training sets
#### So to choose the best, the training set is trained using multiple models( by including polynomial features)
#### The model is applied to the validation set
#### the model which minimizes the error of cross validation set is chosen
#### Inclusion of more features may do well with training set, however may miserably fail on test and validation sets
#### Inclusion of more training data may not necessarily improve the performance
def bias_variance_trade_off(training_x_species_df, training_y_species_df, validation_x_species_df, validation_y_species_df, split_set, species_name,lambda_,alpha,num_iters):
    size_train = training_x_species_df.shape[0]
    end_indices = []
    cost_train_list = []
    cost_validation_list = []
    for i in range(split_set,size_train,split_set):
        end_indices.append(i)
    train_set_df_ = pd.DataFrame()
    for index in end_indices:
        training_x_species_df_ = training_x_species_df[0:index]
        training_y_species_df_ = training_y_species_df[0:index]
        cost_theta_grad_train_ = model_set_optimisation(training_x_species_df_, training_y_species_df_,species_name,lambda_,alpha,num_iters)
        cost_train_ = cost_theta_grad_train_[0][-1]
        theta_ = cost_theta_grad_train_[1]
        cost_validation_list_ = costFunction(validation_x_species_df,theta_[0],validation_y_species_df,0.0)
        cost_validation_ = cost_validation_list_
        cost_train_list.append(cost_train_)
        cost_validation_list.append(cost_validation_)
    plt.plot(end_indices,cost_train_list)
    plt.plot(end_indices,cost_validation_list)
    plt.title("Training error and Cross Validation error" + " " + species_name,fontsize = 12, fontstyle = 'normal')
    plt.xlabel("size of training set",fontsize = 8, fontstyle = 'italic')
    plt.ylabel("Cost J(" + r"$\theta$" + ")",fontsize = 8, fontstyle = 'italic')
    plt.show()

##### The data is plotted to get an idea
def data_visualization(data_species_type_df,species_name):
    plt.suptitle("Iris Dataset",fontsize = 14)
    plt.subplot('321')
    plt.scatter(data_species_type_df[['x1']],data_species_type_df[['x2']],label = species_name)
    plt.xlabel('x1', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x2', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.legend(fontsize = 8)
    plt.subplot('322')
    plt.scatter(data_species_type_df[['x1']],data_species_type_df[['x3']],label = species_name)
    plt.xlabel('x1', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x3', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.legend(fontsize = 8)
    plt.subplot('323')
    plt.scatter(data_species_type_df[['x1']],data_species_type_df[['x4']],label = species_name)
    plt.xlabel('x1', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x4', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.legend(fontsize = 8)
    plt.subplot('324')
    plt.scatter(data_species_type_df[['x2']],data_species_type_df[['x3']],label = species_name)
    plt.xlabel('x2', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x3', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.legend(fontsize = 8)
    plt.subplot('325')
    plt.scatter(data_species_type_df[['x2']],data_species_type_df[['x4']],label = species_name)
    plt.xlabel('x2', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x4', fontsize = 8, fontstyle = 'italic')
    plt.legend(fontsize = 8)
    plt.subplot('326')
    plt.scatter(data_species_type_df[['x3']],data_species_type_df[['x4']],label = species_name)
    plt.xlabel('x3', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.ylabel('x4', fontsize = 8, fontstyle = 'italic', verticalalignment = 'baseline')
    plt.legend(fontsize = 8)

#### The idea is to try different models use it on training set and use it on cross-validation set and choose the one which minimizes itself.
#### Here there are in total 4 features. A total of 18 models can be proposed, restriting to 3 degree polynomials.
def add_poly_features(data_df):
    data_df['x1^2'] = data_df['x1']**2
    data_df['x2^2'] = data_df['x2']**2
    data_df['x3^2'] = data_df['x3']**2
    data_df['x4^2'] = data_df['x4']**2
    data_df['x1x2'] = data_df['x1']*data_df['x2']
    data_df['x1x3'] = data_df['x1']*data_df['x3']
    data_df['x1x4'] = data_df['x1']*data_df['x4']
    data_df['x2x3'] = data_df['x2']*data_df['x3']
    data_df['x2x4'] = data_df['x2']*data_df['x4']
    data_df['x3x4'] = data_df['x3']*data_df['x4']
    data_df['x1^3'] = data_df['x1']**3
    data_df['x2^3'] = data_df['x2']**3
    data_df['x3^3'] = data_df['x3']**3
    data_df['x4^3'] = data_df['x4']**3
    data_df['x1x2x3'] = data_df['x1']*data_df['x2']*data_df['x3']
    data_df['x1x2x4'] = data_df['x1']*data_df['x2']*data_df['x4']
    data_df['x1x3x4'] = data_df['x1']*data_df['x3']*data_df['x4']
    data_df['x2x3x4'] = data_df['x2']*data_df['x3']*data_df['x4']
    train_models_list = [['x0','x1','x2','x3','x4'],['x0','x1','x2','x3','x4','x1^2'],['x0','x1','x2','x3','x4','x1^2','x2^2'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x1^3'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x1^3','x2^3','x3^3'],['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3','x3^3','x4^3'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3','x3^3','x4^3','x1x2x3'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3','x3^3','x4^3','x1x2x3','x1x2x4'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3','x3^3','x4^3','x1x2x3','x1x2x4','x1x3x4'],
    ['x0','x1','x2','x3','x4','x1^2','x2^2','x3^2','x4^2','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','x1^3','x2^3','x3^3','x4^3','x1x2x3','x1x2x4','x1x3x4','x2x3x4']]
    return(data_df,train_models_list)

##### The data is trained for using all the models
##### The best of the model is chosen
##### The model parameters and model which best trains the set is returned 

def train_models(x_data_df, y_data_df, species_name, x_valiadation_df, y_validation_df,lambda_,alpha,num_iters,split_set):
    train_cost_list = []
    theta_list_ = []
    validation_cost_list = []
    difference_train_validation_list = []
    x_data_df_model_list = add_poly_features(x_data_df)
    x_valiadation_df_model_list = add_poly_features(x_valiadation_df)
    for j in range(len(x_data_df_model_list[1])):
#### Choosing the x data features for the model in hand
            x_data_model_df = x_data_df_model_list[0][x_data_df_model_list[1][j]]
            x_valiadation_model_df = x_valiadation_df_model_list[0][x_data_df_model_list[1][j]]
            cost_theta_grad_list_ = model_set_optimisation(x_data_model_df, y_data_df,species_name,lambda_,alpha,num_iters)
            theta_list_.append(cost_theta_grad_list_[1])
            cost_train_model_ = cost_theta_grad_list_[0][-1]
            theta_train_model_ = cost_theta_grad_list_[1]
            cost_validation_model_ = costFunction(x_valiadation_model_df,cost_theta_grad_list_[1][0],y_validation_df,0.0)
            difference_train_validation_ = cost_validation_model_ - cost_train_model_
            train_cost_list.append(cost_train_model_)
            difference_train_validation_list.append(difference_train_validation_)
    plt.plot(range(len(x_data_df_model_list[1])),train_cost_list, label = "Training error", color = 'b')
    plt.xlabel("Models with polynomial features",fontsize = 8, fontstyle = 'italic')
    plt.title("Training error for polynomial features" + " " +species_name,fontsize = 12, fontstyle = 'normal')
    plt.show()
    plt.plot(range(len(x_data_df_model_list[1])),difference_train_validation_list, label = "Cross Validation error - Train error", color = 'r')
    plt.title("Choosing best model fit for" + " " +species_name,fontsize = 12, fontstyle = 'normal')
    plt.xlabel("Models with polynomial features",fontsize = 8, fontstyle = 'italic')
    plt.legend()
    plt.show()
    model_index = indexMin(np.array(difference_train_validation_list))
    title_string_train_error = 'Train test error for'+ " " +species_name
    training_error_list = training_error(x_data_df_model_list[0],theta_list_[model_index][0],x_data_df_model_list[1][model_index],y_data_df,title_string_train_error)
    bias_variance_trade_off(x_data_df_model_list[0],y_data_df, x_valiadation_df_model_list[0], y_validation_df, split_set, species_name,lambda_,alpha,num_iters)
    theta_model_error_list = [theta_list_[model_index][0],x_data_df_model_list[1][model_index],training_error_list]
    return(theta_model_error_list)

##### Since all values are positive and greater than zero, just dividing each column with it's maximum value should work
def feature_scaling(features_data_df):
    column_header_list = list(features_data_df.columns)
    for column in column_header_list:
        if column != 'type':
            max_value_column = features_data_df[column].max()
            features_data_df[column] = features_data_df[column]/max_value_column
    return(features_data_df)


def main():
#### The data is extracted from the .csv file
    iris_data_df = pd.read_csv('../input/iris.csv')
#### Feature feature_scaling
    iris_data_df = feature_scaling(iris_data_df)
### The X (features) for the given dataset is extracted. x0 is = 1 which is bias. There are four variables for this datasets
    iris_data_features_df = iris_data_df[['x0','x1','x2','x3','x4']]
# #### type column of the dataset gives the species to which it belongs to
    y_df = iris_data_df[['type']]
#### this command groups the features corresponding to a particular species
    data_species_types_df = iris_data_df.groupby(['type'])
    theta_all_list = []
    model_all_list = []
    temp_train_set_df = pd.DataFrame()
    temp_validation_set_df = pd.DataFrame()
    temp_test_set_df = pd.DataFrame()
    train_validation_test_sets_list = []
#### Array containing the types of species
    species_type_np = iris_data_df.type.unique()
#### the ratio in which training set, validation set and test set needs to split
    train_set = 0.6
    validation_set = 0.2
    test_set = 0.2
    split_set = 6
#### The data is split into three datasets for each of the species
    for species in species_type_np:
#### data_species_type_df gets the data for a species
        data_species_type_df = data_species_types_df.get_group(species)
        data_visualization(data_species_type_df,species)
    plt.show()
## Iris-setosa values lie outside the domain of the other two species. Visually, for x1 vs x2, for species Iris-versicolor and Iris-virginica
## the values overlap

### the data for the species is split into training, validation and test set
    for species in species_type_np:
        data_species_type_df = data_species_types_df.get_group(species)
        species_sets_dfs_list = extract_sets(train_set,validation_set,test_set,data_species_type_df)
        temp_train_set_df = temp_train_set_df.append(species_sets_dfs_list[0],ignore_index = "True")
        temp_validation_set_df = temp_validation_set_df.append(species_sets_dfs_list[1],ignore_index = "True")
        temp_test_set_df = temp_test_set_df.append(species_sets_dfs_list[2],ignore_index = "True")
#### The training, validation and test sets for all the species are combined
#### train_validation_test_sets_list[0] - training set dataframe for all species - with species name not replaced with 1. A dataframe
#### train_validation_test_sets_list[1] - validation set dataframe for all species - with species name not replaced with 1
#### train_validation_test_sets_list[2] - test set dataframe for all species - with species name not replaced with 1
    train_validation_test_sets_list.append(temp_train_set_df)
    train_validation_test_sets_list.append(temp_validation_set_df)
    train_validation_test_sets_list.append(temp_test_set_df)

    data_x_y_dfs_list = []
#### These parameters have been selected based on bias_variance_trade_off
    lambda_ = [0.3,0.15,1.5]
    alpha = [-0.1,-1.6,-0.1]
    num_iters = [500,800,800]
#### Takes all the sets and splits into three training sets, validation sets and test sets with species name replaced by 1 for the corresponding species
    for set in train_validation_test_sets_list:
        for species in species_type_np:
#### data_x_y_dfs_list[0] - training set for species1
#### data_x_y_dfs_list[1] - training set for species2
#### data_x_y_dfs_list[2] - training set for species3
#### data_x_y_dfs_list[3] - validation set for species1
#### data_x_y_dfs_list[4] - validation set for species2
#### data_x_y_dfs_list[5] - validation set for species3
#### data_x_y_dfs_list[6] - test set for species1
#### data_x_y_dfs_list[7] - test set for species2
#### data_x_y_dfs_list[8] - test set for species3
            data_x_y_dfs_list.append(numeric_conversion_set(set, species, species_type_np))
    for i in range(3):
        theta_model_error_list = train_models(data_x_y_dfs_list[i][0], data_x_y_dfs_list[i][1], data_x_y_dfs_list[i][2], data_x_y_dfs_list[i+3][0], data_x_y_dfs_list[i+3][1],lambda_[i],alpha[i],num_iters[i],split_set)
#### theta_all_list gives the trained theta for the three training datasets - each species
        title_validation_error = "Validation set error for"+ " " +data_x_y_dfs_list[i][2]
        title_test_error = "Test set error for"+ " " +data_x_y_dfs_list[i][2]
        validation_error = training_error(add_poly_features(data_x_y_dfs_list[i+3][0])[0],theta_model_error_list[0],theta_model_error_list[1],data_x_y_dfs_list[i+3][1],title_validation_error)
        test_error = training_error(add_poly_features(data_x_y_dfs_list[i+6][0])[0],theta_model_error_list[0],theta_model_error_list[1],data_x_y_dfs_list[i+6][1],title_test_error)
        print("validation_error")
        print(validation_error)
        print("test_error")
        print(test_error)
        theta_all_list.append(theta_model_error_list[0])
        model_all_list.append(theta_model_error_list[1])
#### From the graphs of the trade_offs it can be seen that the models for the species - Iris-versicolor and virginica are underfitting the data unlike Iris-setosa.
#### It's a high bias problem, where increasing the dataset is not going to be of much help. It needs better representative features to better train the model.

#### predict_validation_species_list is a list, in which each element represents the species predicted for each row in set provied in the input dataframe(validation set in this case)
    predict_validation_species_list = predict_species(add_poly_features(train_validation_test_sets_list[1])[0],theta_all_list,model_all_list)
#### validation_set_y_df is a dataframe which the species are indicated under column 'type'
    validation_set_y_df = train_validation_test_sets_list[1][['type']].replace(to_replace = species_type_np,value = [1,2,3])
#### The maximum value of the predictions is the output. Accuracy is based on it.
    accuracy_validation_set = accuracy_prediction(list(validation_set_y_df['type']), predict_validation_species_list)
    print("accuracy_validation_set")
    print(accuracy_validation_set)
    predict_test_species_list = predict_species(add_poly_features(train_validation_test_sets_list[2])[0],theta_all_list,model_all_list)
    test_set_y_df = train_validation_test_sets_list[2][['type']].replace(to_replace = species_type_np,value = [1,2,3])
    accuracy_test_set = accuracy_prediction(list(test_set_y_df['type']), predict_test_species_list)
    print("accuracy_test_set")
    print(accuracy_test_set)

if __name__ == "__main__":
    main()
