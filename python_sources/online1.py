import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

data = np.genfromtxt('../input/diabetes.csv', delimiter=',')
x = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,stratify=y_train)

def create_model(dropout_rate=0.5, neurons=32):
    model = Sequential()
    model.add(Dense(16,input_dim=x.shape[1],activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons,activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model


model = KerasClassifier(build_fn=create_model,epochs=50,batch_size=50,verbose=0)
param_grid = dict(dropout_rate=[0.2,0.4,0.6],neurons=[20,32,36])

gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv =5, scoring='accuracy')
result = gridSearch.fit(x_val,y_val)
print(result.best_score_)
print(result.best_params_)

model = create_model(dropout_rate=result.best_params_['dropout_rate'],neurons=result.best_params_['neurons'])
model.fit(x_train,y_train,epochs=50,batch_size=50)
y_pred = model.predict_classes(x_test)
print(accuracy_score(y_test,y_pred))
