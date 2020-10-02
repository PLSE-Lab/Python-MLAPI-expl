import tensorflow as tf
print(tf.__version__)
 
import numpy as np
import pandas as pd


credit_card_history= pd.read_csv('../input/creditcard.csv')


print(credit_card_history)

credit_card_shuffled_data = credit_card_history.sample(frac=1) 

one_hot_columns=pd.get_dummies(credit_card_shuffled_data, columns=['Class'] )

normaized_to_small_value= (one_hot_columns - one_hot_columns.min())/ (one_hot_columns.max() - one_hot_columns.min() )

print("normalized value")
print(normaized_to_small_value)
data_features_x = normaized_to_small_value.drop(['Class_0','Class_1'], axis=1)



data_output_y= normaized_to_small_value[['Class_0','Class_1']]


array_data_x= np.asarray(data_features_x, dtype='float32')
array_data_y= np.asarray(data_output_y, dtype='float32')



data_size_for_train = int(0.8 * len(array_data_x))

final_data_trainX , final_data_trainY = array_data_x[:data_size_for_train], array_data_y[:data_size_for_train]



final_data_testX , final_data_testY = array_data_x[data_size_for_train:], array_data_y[data_size_for_train:]


non_fraud_count , fraud_count = np.unique(credit_card_history['Class'], return_counts=True)[1]


#try to give more weightage for fraud case, ie Class_1
# try to avoid and see what is the output
fraud_factor = fraud_count/  (non_fraud_count + fraud_count )
weight_factor= 1/ fraud_factor
#Class_1 column
final_data_trainY[:, 1] = final_data_trainY[:, 1] * weight_factor


# totally 30 features means columns will be given
input_data_features_x = array_data_x.shape[1]

#finally, two outputs Class_0 and Class_1 , output would print in (0 ,1) or (1,0) fomrat
output_data_y = array_data_y.shape[1]

#number of tensors as output or number of cells in layer
layer1_cells_count=150

layer2_cells_count=300


x_train= tf.placeholder(tf.float32, [None,input_data_features_x ], name='x_train')
y_train= tf.placeholder(tf.float32, [None,output_data_y], name='y_train')



weight1 = tf.Variable(tf.zeros([input_data_features_x, layer1_cells_count]), name='weight1')
bias1=  tf.Variable(tf.zeros([layer1_cells_count]), name='bias1')



weight2 = tf.Variable(tf.zeros([layer1_cells_count, layer2_cells_count]), name='weight2')
bias2=  tf.Variable(tf.zeros([layer2_cells_count]), name='bias2')


weight3 = tf.Variable(tf.zeros([layer2_cells_count, output_data_y]), name='weight3')
bias3=  tf.Variable(tf.zeros([output_data_y]), name='bias3')



def train_model(input_values):
    
    output_layre1= tf.nn.sigmoid(tf.matmul(input_values, weight1) + bias1)
    
    output_layre2= tf.nn.sigmoid(tf.matmul(output_layre1, weight2) + bias2)
    
    output_layre2= tf.nn.dropout(output_layre2, 0.5)
    
    output_layre3= tf.nn.softmax(tf.matmul(output_layre2, weight3) + bias3)
    
    return output_layre3
    

y_train_predicted = train_model(x_train)

# put test prediction here


cross_entropy= tf.losses.softmax_cross_entropy( y_train, y_train_predicted)

optimizer= tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)



#add accuracy here and precision as well

with tf.Session() as sess:
    
    variable_init= tf.global_variables_initializer()
    variable_init.run()
  
    for steps in range(200):
        
        opt, cross_ent = sess.run([optimizer, cross_entropy], feed_dict={x_train : final_data_trainX, y_train : final_data_trainY})
        
        
    
    
    
    print("trained the model")
    
    


