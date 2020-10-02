import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint

# Read in train and test data.
train_df = pd.read_csv('../input/optdigits.tra', header=None)
test_df = pd.read_csv('../input/optdigits.tes', header=None)

# The main purpose of the class node is to create a node object that keeps 
# track of it's children.
class Node:
    def __init__(self, df, parent = 'root'):
        self.df = df
        if parent == 'root': # i.e. if the default argument was passed
            self.parent = None
            self.depth = 0
            self.split_column = None
            self.split_threshold = None
            self.child_left = None
            self.child_right = None
        else: # i.e. the node has a parent
            self.parent = parent
            self.depth = parent.depth + 1
            self.split_column = None
            self.split_threshold = None
            self.child_left = None
            self.child_right = None
    
    # return the probabilities of the classes
    def get_class_probabilities(self):
        class_probabilities = []
        if isinstance(self.df, pd.core.series.Series):
            # catch the case when the dataframe is a single row
            class_probabilities.append(1)
        else:
            # if it's a normal dataframe
            row_count = len(self.df.index)
            for i in self.df.iloc[:,-1].unique():
                class_count = sum(self.df.iloc[:, -1] == i)
                class_probabilities.append(class_count / row_count)
        return class_probabilities
    
    # finds the entropy of a node
    def entropy(self):
        entropy_value = 0
        for i in self.get_class_probabilities():
            if i == 0:
                entropy_value += 0
            else:
                entropy_value += -i * np.log2(i)
        return entropy_value
    
    # determines if a node is a leaf by setting a default depth limit, a default probability limit, and a default minimum size.
    def check_leaf(self, max_depth = 4, min_probability = .95, min_size = 20):
        if (len(self.df.index) <= min_size or max(self.get_class_probabilities()) >=\
            min_probability or self.depth >= max_depth):
            return True
        else:
            return False
    
    # gets the value of a leaf
    def output_leaf_value(self):
        return self.df.iloc[:, -1].mode()[0]
    
    # splits the dataframe on a threshold value, default is 4.
    def split_dataframe(self, column, threshold = 4):
        return [ self.df.loc[self.df[column] <= threshold].drop(column, axis = 1), 
                 self.df.loc[self.df[column] > threshold].drop(column, axis = 1) ]
    
    # warns if a leaf is generating children, allows for optimizing the column to split,
    # and splits the dataframe on the column using the default threshold value of 4.
    # returns a list of child nodes.
    def make_child_nodes(self, column = 'optimize' , threshold = 4, max_depth = 10):
        if self.check_leaf(max_depth = max_depth):
            print('Warning: Making child of leaf.')
            
        if column == 'optimize':
            column = self.get_max_info_gain_column(threshold = threshold)
        
        split = self.split_dataframe(column, threshold)
        self.split_column = column
        self.split_threshold = threshold
        self.left_child = Node(split[0], parent = self)
        self.right_child = Node(split[1], parent = self)
        return [self.left_child, self.right_child]

    # computes the entropy of features and the entropy of the entire node, finds the weighted average of new entropies and returns the difference.
    def info_gain(self, column, threshold = 4):
        row_count = len(self.df.index)
        old_entropy = self.entropy()
        new_entropy = 0
        
        for node in self.make_child_nodes(column, threshold):
            proportion = len(node.df.index) / row_count
            new_entropy += proportion * node.entropy()
    
        return old_entropy - new_entropy
    
    # creates a dictionary with info_gain keys and column_name values and returns the column_name with max info_gain
    def get_max_info_gain_column(self, threshold = 4):
        split_options = {}
        for column_name in self.df.columns.values[:-1]:
            split_options[self.info_gain(column = column_name, threshold = threshold)] = column_name
        return split_options[max(split_options.keys())]

# DecisionTree class contains methods for fit, predict, and score
class DecisionTree:
    def __init__(self, root_data_frame, max_depth):
        self.root_node = Node(root_data_frame)
        self.max_depth = max_depth
        self.nodes = [self.root_node]
        self.build_queue = []
        
    # recursively builds decision tree
    def fit(self, current_node = None, threshold = 4):
        if current_node == None:
            current_node = self.root_node
        else:
            pass
        
        if current_node.check_leaf(max_depth = self.max_depth):
            pass
        else: # node is not a leaf, need to expand it and add children to the queue
            children = current_node.make_child_nodes(threshold = threshold)
            for child in children:
                self.nodes.append(child)
                self.build_queue.append(child)
        
        if len(self.build_queue) == 0:
            print('Finished building')
        else:
            current_node = self.build_queue.pop(0)
            return self.fit(current_node = current_node, threshold = threshold)

    # helper function for predict method to make class prediction for input vector by walking down the decision tree         
    def predict_vector(self, vector, current_node = None):
        if current_node == None:
            current_node = self.root_node
        else:
            pass
        
        if current_node.check_leaf(max_depth = self.max_depth):
            return current_node.output_leaf_value()
        else:
            if vector[current_node.split_column] <= current_node.split_threshold:
              return self.predict_vector(vector, current_node = current_node.left_child)
            else:
              return self.predict_vector(vector, current_node = current_node.right_child)
    
    # creates prediction for dataframe by making a prediction for each row
    def predict(self, dataframe):
        # input is dataframe with feature values
        predictions_list = []
        for index, row in dataframe.iterrows():
          predictions_list.append(self.predict_vector(row))
        return pd.Series(predictions_list)
          
    # scores the tree. returns a list with the accuracy table at index 0, the accuracy score at index 1, and the error rate at index 2
    def score(self, test_dataframe):
        predictions = self.predict(test_dataframe)
        labels = test_dataframe.iloc[:,-1]
        accuracy = sum(predictions == labels)/len(predictions)
        label_dict = {i:[] for i in range(10)}
        for i in range(10):
            predicted_i = len(predictions[predictions == i])
            true_positives = sum(labels[predictions == i] == i)
            false_positives = predicted_i - true_positives
            label_dict[i] = [predicted_i, true_positives, false_positives]
        cols = ['Test Cases', 'True', 'False']
        score_df = pd.DataFrame.from_dict(label_dict, orient = 'index', columns = cols)
        err_rate = sum(score_df['False'])/sum(score_df['Test Cases'])
        return [score_df, accuracy, err_rate]

# Printing functions
def print_to_file(score_df, err_rate, percent_train, max_depth, file):
  pprint.pprint('Run with ' + str(percent_train  *100) + '% ' 
                + 'of training set ' + 'and max depth = ' + str(max_depth), file)
  pprint.pprint(score_df, file)
  pprint.pprint('Error Rate: ' + str(round(err_rate,2)), file)
  
def print_to_file2(score_df, run_number, accuracy, percent_train, 
                   max_depth, file):
  pprint.pprint('Run number ' + str(run_number) + ' of 3 for ' 
                + str(percent_train * 100) + '% of data.', file)
  pprint.pprint('Run with ' + str(percent_train  *100) + '% ' 
                + 'of training set ' + 'and max depth = ' + str(max_depth), file)
  pprint.pprint(score_df, file)
  pprint.pprint('Accuracy: ' + str(round(accuracy,2)), file)
  
# This is the logging function. It prints three accuracy tables for various 
# max depths, and five accuracy tables for various sampling proportions.
def log_printer():

  script = open("SCRIPT.txt","w+")
  #max_depth loop
  for i in [3, 5, 7]:
    decisiontree = DecisionTree(train_df, max_depth = i)
    decisiontree.fit()
    score_info = decisiontree.score(test_df)
    score_df = score_info[0]
    err_rate = score_info[2]
    print_to_file(score_df, err_rate, 1, i, script)

  #random sample loop
  sample_props = [.1, .2, .5, .8, 1]
  for i in sample_props:
    trials = []
    for j in range(1,4):
      decisiontree = DecisionTree(train_df.sample(frac=i), max_depth = 7)
      decisiontree.fit()
      score_info = decisiontree.score(test_df)
      score_df = score_info[0]
      accuracy = score_info[1]
      trials.append(accuracy)
      print_to_file2(score_df, j, accuracy, i, 7, script)
    pprint.pprint('Mean accuracy over 3 trials for sample size ' + str(i) + 
                 ' = ' + str(np.mean(trials)), script)

  script.close()

def chart_maker():
    # makes a graph with sample proportions on the x-axis and accuracies on the y-axis
    sample_props = [.1, .2, .5, .8, 1]
    accuracy_vals = []
    for i in sample_props:
        trials = []
        for j in range(3):
            decisiontree = DecisionTree(train_df.sample(frac=i), max_depth = 7)
            decisiontree.fit()
            score_info = decisiontree.score(test_df)
            accuracy = score_info[1]
            trials.append(accuracy)
        accuracy_vals.append(np.mean(trials))

    training_examples_accuracy = dict(zip(sample_props, accuracy_vals))
    
    plt.scatter(training_examples_accuracy.keys(), training_examples_accuracy.values())
    plt.xlabel('Sample Proportion')
    plt.ylabel('Accuracy')
    plt.savefig('training_examples_accuracy.png')

def chart_maker_depth():
    # makes a chart with depths on x-axis and accuracy on y-axis
    depths = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    accuracy_vals = []
    for i in depths:
        decisiontree = DecisionTree(train_df, max_depth = i)
        decisiontree.fit()
        score_info = decisiontree.score(test_df)
        accuracy = score_info[1]
        accuracy_vals.append(accuracy)

    training_examples_accuracy = dict(zip(depths, accuracy_vals))
    
    plt.scatter(training_examples_accuracy.keys(), training_examples_accuracy.values())
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.savefig('training_examples_accuracy.png')
    

# chart_maker_depth()
log_printer()