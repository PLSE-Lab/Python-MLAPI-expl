# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

print('Loading training set...')
dataset = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
y_test = test_data.values[:, :]
X_train = dataset.values[:, 1:]
y_train = dataset.values[:, 0]

print('Start Learning...')
recognizer = RandomForestClassifier(n_estimators = 150, criterion = 'entropy')
recognizer.fit(X_train, y_train)

y_pred = recognizer.predict(y_test) #answer of test data :) :) 

plot_test = y_test[0:10, :] 
# Plotting and confirming few results
for i in plot_test:
    pixels = i.reshape((28, 28))
    plt.imshow(pixels, cmap='gray') 
    plt.show()
    
df = pd.DataFrame({'ImageId': pd.Series(range(1, 28001)) , 'Label': y_pred})
df.to_csv('Output.csv', index = False)