import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

np.random.seed(1029)
# Read CSV
train_type = {'Id': 'str',
        'Cone Latitude': 'float64',
        'Cone Longitude': 'float64',
        'Distance': 'float64'}

df_train = pd.read_csv('training.csv', dtype=train_type)


# Load images into array
train_image = []
img_width, img_height = 960, 540
# 960,540
for png in tqdm(range(df_train.shape[0])):
    img = image.load_img('TrainingImages/'+df_train['Id'][png], target_size=(img_width, img_height, 3), color_mode='rgb')
    img = image.img_to_array(img)
    img = img.flatten()
    # print(img.shape)
    train_image.append(img)
    img = []
# get data points into individual arrays / Series'
x = np.array(train_image)


lat = df_train['Cone Latitude'].values
long = df_train['Cone Longitude'].values
dist = df_train['Distance'].values
# Split Train data to 75/25
x_train, x_test, lat_train, lat_test, long_train, long_test, dist_train, dist_test = train_test_split(x, lat, long, dist, random_state=1029, test_size=.5)
print(dist_train)

tree = DecisionTreeRegressor()
tree = tree.fit(x_train, dist_train)
test_pred = tree.predict(x_test)

df_validation = pd.read_csv('sample.csv')
validation_image = []

for png in tqdm(range(df_validation.shape[0])):
    img = image.load_img('TestingImages/'+df_validation['Id'][png], target_size=(img_width, img_height, 3), color_mode='rgb')
    img = image.img_to_array(img)
    img = img.flatten()
    validation_image.append(img)

x_val = np.array(validation_image)
df_validation.Distance = tree.predict(x_val)

df_validation.to_csv('sample1.csv', index=False)