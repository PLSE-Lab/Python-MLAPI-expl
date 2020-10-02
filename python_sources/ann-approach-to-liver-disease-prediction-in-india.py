import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Import Dataset
dataset = pd.read_csv('../input/indian_liver_patient.csv')

# Split into matrix of features and Matrix of Dependent Variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encode categorical varaibles
label_encoder_X = LabelEncoder()
X[:,1] = label_encoder_X.fit_transform(X[:,1])
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Scale Data
st_sc = StandardScaler()
X = st_sc.fit_transform(X)

# Split X and y into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Create Model
model = Sequential()

model.add(Dense(32, input_dim = 10, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 64, epochs = 500, verbose = 0)

score = model.evaluate(X_test, y_test)
print(score)