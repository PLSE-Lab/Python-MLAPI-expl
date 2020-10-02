#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 
# <h1><span style ="font-family:charcoal;font-size:32px;color:Blue;;"><b>   Visual Guide to Convolution Neural Networks (CNNs)&nbsp; <span style="color:Tomato;" >FOR EVERYONE&nbsp;</span></b></span></h1>

# <img src="https://thumbs.gfycat.com/HelpfulScratchyArcticseal.webp">

# <h1 style="color:Tomato;"><b><i>INTRODUCTION</i></b></h1>  

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761">"How do we actually identify the objects we see?" this is one of the basic questions we have to ask ourselves before we try to implement it as CNNs. to answer this question let us look at an example. </h4>
# 

# <img src="https://upload.wikimedia.org/wikipedia/commons/0/09/TheCheethcat.jpg" width="300" height="400">

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><ul><li>Above is the picture of a Cheetah , clearly , but how do we identify it as a cheetah ? it is done by noticing the various&nbsp;<i>features</i>&nbsp; that the cheetah posseses , such as it's colour , the spots on its coat , the 'tear lines' that run in between its face etc.</li><li>Thus we understand that , objects are classified/identified by us based on its distinctive characteristics /<i>&nbsp;features</i>&nbsp;.</li><li>the result of implementing this concept of identifiying objects with the help of it's characteristics is what lead to the development of <span style ="color:Tomato"><i>&nbsp;&nbsp;Convolution Neural Networks (CNNs)</i></span></li><li>Just like how our eyes scan through an image to look for features , CNNs scan through through the image to find various features. </li><li>Although the method is same , the features that are identified by CNNs maybe different than what humans use to classify an object . this is what causes mis-classification , but in some cases this behaviour also allows models to identify objects from images that even humans cannot easily interpret!</li><li>if u think about it , this behaviour can also be found among different people , it is not that all of us might interpret an object the same way, everyone thinks differently and hence we might have a difference in opinion , given below are some examples that will help u understand what I'm talking about</li></ul> </h4>

# <img src="https://i.pinimg.com/236x/68/72/83/6872834a4d21c9ea4e332deb16faa9db.jpg" width="300" height="400">

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><p>what do u see in the above image? ;)</p>
# <p>This is the famous old woman / young lady illusion . The  image can be interpreted in to ways :
# <ol><li>the young lady : there is a young lady with a sharp jawline and curly brown short hair looking into the screen , u can see her small nose , next to her eyleash ,u can see her left ear ,  and u see that she is wearing a pinkish necklace neckless . </li><li>The Old Lady : what appeared to be the young lady's jawline is now the nostrill outline of the old woman's nose, who is facing downwards ,  with her left eye closed (the left eye appears as the young lady's left ear ). U can see her thin lips ( which appeared as the young lady's necklace ) . she has an overall 'witchy' appearance . the little wart on her nose is what appeared as the young lady's nose above which u can see an eyelash.</li></ol></p><p>It might take a while to notice both the old woman and the young lady . from this example , we can see how focussing on different features leads to different classification of the same image . Now that we have seen the importance of features to classify the image. let us move on to <i>how it is done is CNNs</i></p> 
#     

# <p><h1 style="color:Tomato;"><b><i>WHAT IS AN IMAGE ? (how does a computer comprehend it?)</i></b></h1></p>
# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"><ul><li>Images are basically , either a 2d array (row x column) of values from 0-255 for a gray-scale image or a 3d  array (row x column x channel)  of values from 0-255 for a colour image.<br><br> </li><li>the 3 dimentions are : <br><br> <ol><li> rows of pixels </li><li>columns of pixels</li><li>(if coloured) how red/green/blue an image is respectively (all colours are intricate combinations of brightnesses of the  three basic colours red green and blue).</li></ol><br><li>the values (0-255) corresponds to the brightness of that particular pixel . </li></ul><br>
# <img src="https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png" width="500" height="600"></h4>
# 
# 

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:18px;color:#334761"> let us now move on to understanding the elements of CNNs</h4>

# <p><h1 style="color:Tomato;"><b><i>ELEMENTS OF CNNs : </i></b></h1></p>
# <P>
# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><ol><li><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>FEATURE DETECTORS</i></b></span> &nbsp;&nbsp;(aka filters/kernels/array of weights) &nbsp; : <br><br>
#     <img src="https://media1.giphy.com/media/2kXLNQypdX9O1A3zxX/giphy.gif" width="400" height="300">
#     <br><br><ul><li><span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>FEATURE DETECTORS </i></b></span>&nbsp;&nbsp; are small arrays ( usually of dimentions 3x3) that contain values that act as  multipliers . the pixel values of the image are changed when filters are applied to it . This new image with changed pixel values will reveal different features based on the filter that is applied .<br><br></li><li>how is a filter applied ? : <br><br><ul><li>the filter is slided accross an image in a step by step fashion . each step is called a stride . At each stride ,the values of the sub-array of the image corresponding to the position of the filter is  element-wise multiplied to the values of the filter , and these new values are summed up to give one single final value . </li><li> the resultant value of each stride/step is put into a new array called the <b><i>feature map . The process of how this feature map is formed is called <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>CONVOLUTION</i></b></span></i></b></li><li>the resultant matrix (feature map) is a compressed form of the image reducing the rows and column dimention by a value of 2 , that is if a 3x3 filter with stride = 1(row/columns increasing by 1 step) is moved along a 15x15 image , the feature map will be of size 13x13 . if we want a feature map of the same dimention as the input image, we can add the required amount of padding (columns / rows of 0s). </ul><br><br><br>
#     <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:25px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>2D CONVOLUTION VISUAL (step by step) (without padding)</i></b></span>
#     <br><br><br>
#     <img src="https://aigeekprogrammer.com/wp-content/uploads/2019/12/CNN-filter-animation-1.gif" width="900" height="800">
#     <br><br><br>
#     </h4>
#     <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:25px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>3D CONVOLUTION VISUAL (step by step)</i></b></span>
#     <br><br><br>
#     <img src="https://www.researchgate.net/profile/Santhiya_Rajan/post/How_will_channels_RGB_effect_convolutional_neural_network/attachment/5c67b72d3843b0544e664e12/AS%3A726829115666434%401550300973344/download/cnn_1.gif" width="900" height="800">
#     <br><br><br>
#     </h4>
#     <li>One part of CNNs are groups of such filters , these groups of filters are called<span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>CONVOLUTION LAYERS</i></b></span></li>
#     <li>below is a visual representation of how feature maps of each [filter / array of weights] is stacked .
#         <br><br><br>
#     <img src="https://i.stack.imgur.com/FjvuN.gif" width="800" height="700">
#     <br><br><br>
#    </li><li>Now that we have understood what feature detectors are and what the Convulution layer is  , we will move on to the next element of CNNs</li></ul><br><br><br>           <li> <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>RELU function :</i></b></span>&nbsp;&nbsp;<br><br><br>
#     <img src="https://machinelearningknowledge.ai/wp-content/uploads/2019/08/Relu-Activation-Function.gif" width="600" height="500">
#     <br><br><ul><li>
#     <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>RELU FUNCTION</i></b></span>&nbsp;&nbsp;(REctifier Linear Unit) : it is the most common activation function used in Neural Networks .</li><li>WHY RELU ? : <ul><li>images are genrally non linear by nature , this non linearity should be maintained to keep the performance of CNN optimal .</li><li> when convolution is done , there is a risk of linearity in the feature map , this is where RELU comes in ,  using the RELU as the activation function ensures that this non linearity is maintatined . </li> </ul></li></ul><br><br>
#     <li><h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>POOLING :</i></b></span>&nbsp;&nbsp;<br><br><br>
#     <img src="https://media1.tenor.com/images/f181464c1be3f16db829c46966eab6fd/tenor.gif?itemid=3452012" width="400" height="300">
#     <br><br>
#         <ul><li>WHAT IS POOLING ? AND HOW IS IT USEFULL ? : <br><br><ul><li>Pooling is a technique that is used to controll overfitting , by providing <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>SPATIAL INVARIANCE</i></b></span>&nbsp;&nbsp;and compressing the data further . ( keeps the values that indicate the positions of the features and removes rest of the values )</li><li><span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>SPATIAL INVARIANCE : </i></b></span>&nbsp;&nbsp; it is a property that allows the CNN to detect features even if they are slightly displaced or disoriented .</li><li>By compressing the data without loosing much information , it reduces the computional load on CNNs.</li></ul>
#             <br><br><li>HOW IS POOLING DONE ? : <br><br><ul><li>Like Convolution , pooling is also done by striding along the feature maps. </li><li>In each stride , a single value is calculated that will replace the values of that particular patch in that stride</li><li>The value calculated is based on the type of pooling that is done .</li><li> the resultant values are packed into an array just like when feature maps were made . this process is continuoes for 'n' feature maps resulting in 'n' pooled feature maps .</li><li>This layer that does the pooling function is called the <i> pooling layer </i> </li><li> The most common and preffered type of pooling is <span style="font-size:20px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>MAX POOLING</i></b></span>&nbsp;&nbsp;(aka down-sampling). <br><span style="font-size:20px;color:#334761;color:Tomato"><b><i>MAX POOLING</i></b></span> &nbsp;&nbsp;:  for each stride the maximum of the values belonging to that particular patch is selected . hence the pooled feature map will contain the maximum values of all the the patches . Larger values indicate the presence of features , hence by keeping the maximum values , we compress the feature map without loosing inportant information . </li><li>other types of pooling techniques are : <br><br><ul><li><i>AVERAGE POOLING</i>&nbsp;&nbsp;(aka sub-sampling) : pooling the feature map by calculating the average of values of each patch of the featre map .</li><li><i>GLOBAL POOLING</i>&nbsp;&nbsp;: special case of down-sampling where [stride/patch] size = size of feature map , That is , the feature map is reduced to a single value .</li>
#     <li><i>SUM POOLING</i>&nbsp;&nbsp;:the sum of all the values of each patch of the feature map is calculated to make the pooled feature map. <br><br><br>
#     <img src="https://www.bouvet.no/bouvet-deler/understanding-convolutional-neural-networks-part-1/_/attachment/inline/e60e56a6-8bcd-4b61-880d-7c621e2cb1d5:6595a68471ed37621734130ca2cb7997a1502a2b/Pooling.gif" width="900" height="800">
#     <br><br>
#             </li></ul></li></ul></li></ul>
#         <ul><li>
#             Till now we have come across convolution layers , the relu function and pooling .</li><li>During application , the relu function is built into the convolution layers . </li><li>we add multiple convolutional and pooling layers , to extract more advanced features and reduce the data to a very efficiently compressed format . one step remains in the convolutional neural network and that is the ... <br><br>  
#             </li></ul> <li> <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:30px;color:#334761;color:Green">&nbsp;&nbsp;<b><i>FLATTENING LAYER :</i></b></span>&nbsp;&nbsp;<br><br><br>
#     <img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/b4b71c37-c0e7-45b1-9b28-dd1e52c99fb9/d8hlsgu-9ab3e032-ecfa-4832-9498-d20c54ba9a19.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3sicGF0aCI6IlwvZlwvYjRiNzFjMzctYzBlNy00NWIxLTliMjgtZGQxZTUyYzk5ZmI5XC9kOGhsc2d1LTlhYjNlMDMyLWVjZmEtNDgzMi05NDk4LWQyMGM1NGJhOWExOS5naWYifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.hI41Awsbr9E7m1QKJhpRrCf3Cn65695bTFwbMh9zPFQ" width="400" height="300">
#     <br><br> <span style="font-size:20px;color:#334761;color:Tomato"><b><i>FLATTENING LAYER</i></b></span>&nbsp;&nbsp;:<br><br> <ul><li>Now that we have the data in a very compressed format , we want to convert it in such a way that it can be passed as input to a machine learning model (prefferably a Artificial Neural Network ) . </li><li>The flattening layer completes this task by reshaping and concatenating each of the final feature maps into a looong vector as shown in the image below.
#         <br><br>
#     <img src="https://www.researchgate.net/profile/Budiman_Minasny/publication/334783857/figure/fig4/AS:786596169269249@1564550549811/Illustration-of-flatten-layer-that-is-connecting-the-pooling-layers-to-the-fully.png" width="600" height="500">
#         <br><br></li></ul>
# 
#  <span style="font-size:20px;color:#334761;color:Tomato"><b><i>NOTE  </i></b></span>&nbsp;&nbsp;: just like we preprocess data in any machine learning problem , we need to preprocess our data for CNNs , this is done by normalizing the pixel values of the image so that it lies from 0 to 1. this helps the model's complex calculations and in reducing computational load , hence decreases the training time .  <br><br>
#         
#  And finally ...  
# <br><br><li><span style="font-size:20px;color:#334761;color:Tomato"><b><i>PREDICTION MODELLING</i></b></span>&nbsp;&nbsp;: artifcial neural networks are the most common models used to make predictions taking the flattened vector as input . </li></ol>
# 
# <br> <span style="font-size:20px;color:#334761;color:Tomato"><b><i>Some important terms </i></b></span>&nbsp;&nbsp;:
# <ul><li><span style="font-size:18px;color:#334761;color:Green"><b><i>SOFTMAX FUNCTION</i></b></span>&nbsp;&nbsp;: it is the activation function of the final predictive layer of a multi-classification CNN model . it predicts the probabiliy of the image belonging to each of the classes . the class with the highest probability is considered as the prediction.</li>
#     <li><span style="font-size:18px;color:#334761;color:Green"><b><i>CATEGORICAL CROSS-ENTROPY</i></b></span>&nbsp;&nbsp;: It is one of the most preffered loss function used for single label classification in CNNs .
#         <br><br><br></li></ul></li>
# We will bring all the different elements together and analyse the complete framework of a CNN now . 
#     <img src="https://static.wixstatic.com/media/d77a32_076f341258d24f47852625faaa726620~mv2.jpeg/v1/fill/w_1422,h_760,al_c,q_85,usm_0.66_1.00_0.01/1_uAeANQIOQPqWZnnuH-VEyw.webp" width="1200" height="1100">
#         <br><br>
# <span style="font-size:18px;color:#334761;color:Green"><b><i>FRAMEWORK DETAILS </i></b></span>&nbsp;&nbsp;: The above framework contains 2 convolution layer [conv_1 , conv_2] , each of these layers followed by a pooling layer . it contains an artificial NN for predictive modelling that takes the final flattened vector as input .
#     <br><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>input layer</i></b></span>&nbsp;&nbsp;: a gray-scale image of dimentions [28x28x1]<br><br></li><li><span style="font-size:18px;color:#334761;color:Green"><b><i>conv_1</i></b></span>&nbsp;&nbsp;: contains 'n1' filters , each of dimention 5x5 , stride_steps (row/column movement per stride) = 1 . Converts the input layer in n1 [24x24] sized feature maps since it has n1 filters.</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>conv_2</i></b></span>&nbsp;&nbsp;: contains ['n2'/'n1'] filters , each of dimention 5x5 , stride_steps = 1 .Converts the output of pooled-conv_1 into n2 [8x8] sized feature maps since it has n2/n1 filters ( each of n1 filters are convoluted into [n2/n1] feature maps .</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>max pooling layer</i></b></span>&nbsp;&nbsp;:patch size = 2x2 , stride_steps = 2 .reduces size of the feature maps by 75%.</li><br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>flattening layer</i></b></span>&nbsp;&nbsp;: converts the final 'n2' feature maps of dimention 4x4 , into a single single vector of length (4 x 4 x n2).</li>
# <br><li><span style="font-size:18px;color:#334761;color:Green"><b><i>predicive modelling : </i></b></span>&nbsp;&nbsp; <ul style="list-style-type:circle; margin-left: 10px;" ><br><li>input layer : flattened vector </li><li> hidden layer of n3 nodes </li><li> output layer of 10 node (for classes [0 to 10]) </ul>
#     <br> 
#    
# <span style="font-size:30px;color:#334761;color:Green"><b><i>As of now we have succesfully learnt the basics of CNNs . </i></b></span>&nbsp;&nbsp;<span style='font-size:100px;'>&#129346;</span></h4>
# 
#  
# 
# 
# 

# <h1  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:Blue"> It is finally time for us to move on the application part of CNNs 

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761">
# <span style="font-size:38px;color:#663399;font-family:Monaco,sans-serif;">MNIST classifcation </span>&nbsp;&nbsp;
#     <br><br><p>We will use the architecture shown in the framework above for easy reference and understanding.</p>
# 

# 
# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">UNDERSTANDING THE DATA </span>&nbsp;&nbsp;:<br><br><ul><li>The training data set, (train.csv), has 785 columns. </li><li>The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.</li><li>
# Each pixel column in the training set has a name like pixel 'x', where 'x' is an integer between 0 and 783, inclusive. </li><li>To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).</li><li>The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.</li><li>
# Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. </li><br><b>format :<br><br>ImageId,Label<br></b>
# 1,3<br>
# 2,7<br>
# 3,8 <br>    
#     (27997 more lines)</li></ul><br><br>
#  <span style="font-size:28px;color:Green;font-family:Monaco,sans-serif;">let us begin by importing the necassary libraries , we will use keras implementation of CNNs in this notebook .</span>&nbsp;&nbsp;
#  
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D
from keras.models import Model
from keras.layers import Input,InputLayer
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import h5py


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><ol><li><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">DATA PREPROCESSING </span>&nbsp;&nbsp;:
#     <br><br><ol><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">ISOLATING THE LABELS FOR EACH IMAGE </span>&nbsp;&nbsp; :<br>since the first column contains the target (label of the image) . we will isolate it from the training data. we will also one-hot-encode the label beacuse this is the desired format of labels for keras when softmax function is used from predictions .</li><br><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">RESHAPING THE INPUT INTO A KERAS ACCEPTABLE FORMAT </span>&nbsp;&nbsp; :<br> we will transform the input into an array of dimention [length of image x breadth of image x no.of images] which is acceptable by the keras implementation.<br><br><li><span style="font-size:24px;color:green;9;font-family:Monaco,sans-serif;">SCALING</span>&nbsp;&nbsp; :<br>we will divide all the pixel values by 255 so that all the pixel values will lie from [0-1] .</li></ol>
# <br>We will now implement data preprocessing using the function data_procc()
# 
#    
#     
# 

# In[ ]:


def prep_data(raw):
    img_rows, img_cols = 28, 28
    num_classes = 10
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y , img_rows, img_cols, num_classes, num_images

mnist_file = "/kaggle/input/digit-recognizer/train.csv"
mnist_data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')
x, y, img_rows, img_cols, num_classes, num_images= prep_data(mnist_data)
print('input shape : {}'.format(x.shape))
print('output shape : {}'.format(y.shape))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:#334761;9;font-family:Monaco,sans-serif;">now that we have completed data preprocessing we will move on to buliding out model .</span>&nbsp;&nbsp;
#     <br><br><br>
#     since we decided our model will be the same as the framework we have seen before , let us look at it again .
#     <br><br><br>
#         <img src="https://static.wixstatic.com/media/d77a32_076f341258d24f47852625faaa726620~mv2.jpeg/v1/fill/w_1422,h_760,al_c,q_85,usm_0.66_1.00_0.01/1_uAeANQIOQPqWZnnuH-VEyw.webp" width="1200" height="1100">

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:38px;color:orange;9;font-family:Monaco,sans-serif;">ARCHITECTURE</span>&nbsp;&nbsp;: <br><br><br><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;"><br><br><br>1st convolutional layer</span>&nbsp;&nbsp;: <br><br> <ul><li>no of filters (n1) = 16<li>filter dimentions = [5x5]</li><li>we will use the relu function as the activation function for all the layers except the output layer.</li><li>we need to pass the dimentions of the input as input shape , only for the first layer</li>   

# In[ ]:


#initializing our model
cnn_pilot = Sequential()

#adding the first convolutional layer
cnn_pilot.add(Conv2D(filters = 16,
                     kernel_size = 5,
                     activation = 'relu',
                     input_shape = (img_rows, img_cols, 1)))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">1st pooling layer</span>&nbsp;&nbsp;: <br><br>the pooling layer takes patches of size [2x2] into consideration and has a stride of size = 2 as default . </h4>

# In[ ]:


#adding the first pooling layer
cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">2nd convolutional layer</span>&nbsp;&nbsp;: <br><br> <ul><li>no of filters (n2) = 32<li>filter dimentions = [5x5]</li><li>we will use the relu function as the activation function.</li></ul>  </h4>

# In[ ]:


#adding the second convolutional layer
cnn_pilot.add(Conv2D(filters = 32,
                     kernel_size = 5,
                     activation = 'relu'))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">2nd pooling layer</span>&nbsp;&nbsp;: <br><br>the 2nd pooling has same specifications as the first pooling layer.</h4>

# In[ ]:


#adding the 2nd pooling layer
cnn_pilot.add(MaxPooling2D(pool_size=(2,2)))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">flattening layer</span>&nbsp;&nbsp;: <br><br>we will now add the flattening layer</h4>

# In[ ]:


#adding the flattening layer , whose output will be used as the final NN's input
cnn_pilot.add(Flatten())


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:20px;color:#334761"><span style="font-size:28px;color:green;9;font-family:Monaco,sans-serif;">final neural network (prediction model):  </span><br><br><ul><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">input layer :</span>&nbsp;&nbsp;the input layer for the final Neural Network is the same as the flattening layer's output, that is the flattened vector.<br><br><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">hidden layer :</span>&nbsp;&nbsp;hidden layer of n3 = 64 nodes and activation funtion = relu<br><br><li><span style="font-size:28px;color:limegreen;9;font-family:Monaco,sans-serif;">output layer :</span>&nbsp;&nbsp;nodes = number of classes(10) and activation funtion = softmax </h4>

# In[ ]:


#adding the hidden layer
cnn_pilot.add(Dense(units= 64,
                    activation='relu'))
cnn_pilot.add(Dense(units = num_classes,
                    activation = 'softmax'))


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">The last thing that we need to do is 'compile' our CNN model . This is where we specify the loss-function (we will use 'categorical-crossentropy') , the optimizer (we use 'adam' so that the model's learning rate is automatically optimized ) , and for easy evaluation , the metric we will choose is accuracy .</h4>

# In[ ]:


cnn_pilot.compile(loss = "categorical_crossentropy",
                   optimizer = 'adam',
                   metrics = ['accuracy'])


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Let us now check our final CNN model</h4>

# In[ ]:


cnn_pilot.summary()
Wsave = cnn_pilot.get_weights()


# In[ ]:


print(Wsave)


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Looks like all the layers of the CNN are in place .<br>We have finally finished building our very own Convulution Neural Network!!  <span style='font-size:100px;'>&#127870;</span>      <br><br>
#     <img src="https://media1.tenor.com/images/5f43741cbdfb89e277afbfe4fa3b0cbc/tenor.gif?itemid=10683543" width="600" height="500">
#         <br><br></h4>

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">Let us now fit our model to the data and see it in action.</h4>

# In[ ]:


history = cnn_pilot.fit(x,
              y,
              batch_size = 100,
              epochs = 100,
              validation_split = 0.2 )


# In[ ]:


print(Wsave)


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">To better analyse our data , let us plot its performance as a function of no.of epochs.</h4>

# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('loss')
plt.legend()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('accuracy')
plt.legend()
plt.show()


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">We see that after a certain no.of epochs the validation loss increases and from there on osciliates non-unformly . this means that beyond a certain epoch our model begins to overfit ! . One of the methods of controlling overfitting is by using <span style="font-size:25px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>'early stopping' . </i></b></span><br><br><span style="font-size:25px;color:#334761;color:Tomato">&nbsp;&nbsp;<b><i>early stopping</i></b></span>&nbsp;&nbsp;: while the model trains , beyond a certain epoch the loss and accuracy tends to fluctuate. we try to 'catch' the model when it is at its best by saving the model when it has it's lowest loss or highest accuracy , some allowance is given to ensure that the we have the best model , that is called 'patience'. patience is the no of epochs that is allowed to train , to check if the accuracy/loss gets better or not .<br><br>(credits to <a href="https://www.kaggle.com/rameshbabugonegandla">@ramesh</a> for this helpfull suggestion)

# In[ ]:


cnn_pilot.set_weights(Wsave)

es = EarlyStopping(monitor='val_loss', mode='min', verbose= 1 , patience=5)
mc = ModelCheckpoint("best_cnn.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = cnn_pilot.fit(x,
                        y,
                        batch_size = 100,
                        epochs = 100,
                        validation_split = 0.2,
                        verbose = 1,
                        callbacks = [es,mc])


# In[ ]:


cnn_best = load_model("best_cnn.h5")
_, train_acc = cnn_best.evaluate(x, y, verbose=0)
print("Train: {}".format(train_acc))
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('loss')
plt.legend()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('accuracy')
plt.legend()
plt.show()


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761"> we see that we caught the model before it tends to overfit.

# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">the working of a CNN might seem a little 'blackboxy' , so to better our intuition we will visualize the journey of an image through the CNN . enjoy the ride ;)</h4>

# In[ ]:


def image_journey(img_num):
    layer_outputs = [layer.output for layer in cnn_pilot.layers[:4]]
    activation_model = keras.Model(inputs=cnn_pilot.inputs, outputs=layer_outputs)
    activations = activation_model.predict(x[img_num].reshape(1,28,28,1))
    layer_list=['1st convolution layer','1st pooling layer','2nd convolution layer','2nd pooling layer']
    fig, ax = plt.subplots(1, 1 ,figsize = (4,4))
    plt.subplots_adjust(left=0, bottom=-0.2, right=1, top=0.9,wspace=None, hspace=0.1)
    fig.suptitle('original image', fontsize=30)
    ax.imshow(x[img_num][:,:,0], cmap='gray')
    for i in range(4):
        activation = activations[i]
        activation_index=0
        fig, ax = plt.subplots(1, 6 ,figsize = (30,3))
        fig.suptitle(layer_list[i], fontsize=50)
        plt.subplots_adjust(left=0, bottom=-0.6, right=1, top=0.6,wspace=None, hspace=0.1)
        for row in range(0,6):
            ax[row].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
image_journey(200)


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761">we finally reached the last part of the notebook , where we will submit our predictions of the test data . <br><br>Just as we pre processed our training data we will preprocess the test data , make our predictions with our CNN model and submit our predictions.</h4>

# In[ ]:


mnist_test = "/kaggle/input/digit-recognizer/test.csv"
mnist_test = np.loadtxt(mnist_test, skiprows=1, delimiter=',')
num_images = mnist_test.shape[0]
out_x = mnist_test.reshape(num_images, img_rows, img_cols, 1)
out_x = out_x / 255
results = cnn_pilot.predict(out_x)
results = np.argmax(results,axis = 1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})
submissions.to_csv("submission.csv", index=False, header=True)


# <h4  style="font-family:Tahoma, Geneva, sans-serif;font-size:25px;color:#334761"><ul><li>Thank you for reading till the end of this notebook .</li><li>Self learning is a very frustrating process , so my agenda is to make fellow learner's journey as easy as possible through my notebooks . </li><li>if you liked this notebook , please <span style="font-size:28px;color:orange;9;font-family:Monaco,sans-serif;">upvote</span>&nbsp;it to help me upload more engaging and intuitive notebooks . </li><li>cheers! and happy machine learing.<span style='font-size:80px;'>&#127867;</span></h4>

# In[ ]:




