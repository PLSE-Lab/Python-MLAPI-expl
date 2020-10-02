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

def create_model(first_neurons=16,second_neurons=32):
    model = Sequential()
    model.add(Dense(first_neurons,input_dim=x.shape[1],activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(second_neurons,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model


model = KerasClassifier(build_fn=create_model,epochs=50,batch_size=50,verbose=0)
param_grid = dict(first_neurons=[12,16,24],second_neurons=[30,40,50])

gridSearch = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='precision')
gridSearch.fit(x_val,y_val)

print(gridSearch.best_score_)
print(gridSearch.best_params_)

model = create_model(first_neurons=gridSearch.best_params_['first_neurons'], second_neurons=gridSearch.best_params_['second_neurons'])
model.fit(x_train,y_train,epochs=50,batch_size=50)
y_pred = model.predict_classes(x_test)
print(accuracy_score(y_test,y_pred))

    
