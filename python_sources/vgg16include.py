import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation
from keras.models import Sequential

# Initialize the model
model = VGG16(weights='imagenet', include_top=True)

# Get the summary of the model
print(model.summary())

# Get the total layers in the model
print("No of layers in the model : " + str(len(model.layers)))