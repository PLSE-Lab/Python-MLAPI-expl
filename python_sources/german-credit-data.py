
# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %% [code]
cred= pd.read_csv('../input/German_Credit_data.csv')

# %% [code]
cred.head()

# %% [code]
len(cred)

# %% [code]
x=cred.iloc[:,1:22].values

# %% [code]
y=cred.iloc[:,0].values

# %% [code]
x

# %% [code]
y

# %% [code]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

# %% [code]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, random_state=100)

# %% [code]
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(x_train,y_train)

# %% [code]
y_prd = lm.predict(x_test)

# %% [code]
y_prd

# %% [code]
y_test

# %% [code]
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_prd)

# %% [code]
accuracy = (31+159)/(31+37+23+159)

# %% [code]
accuracy

# %% [code]
from sklearn.metrics import classification_report

# %% [code]
print(classification_report(y_test,y_prd))

# %% [code]
from sklearn.metrics import roc_auc_score

# Let us measure the Logistic Regression model by the help of ROC and AUC curve

logistic_roc_auc = roc_auc_score(y_test,y_prd)
print(logistic_roc_auc)

# %% [code]
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,y_prd)

# %% [code]

display (thresholds[:10])
display (fpr[:10])
display (tpr[:10])

# %% [code]
lm.predict_proba(x_test)[:,1][:5]

# %% [code]
#Visualisation for ROC and AUC Curve
import matplotlib.pyplot as plt
%matplotlib inline


# Plot of ROC Curve for specific class

plt.figure()
plt.plot(fpr, tpr, label = 'ROC Curve (area= %0.2f)' %logistic_roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Exampls')
plt.legend(loc='lower right')
plt.show()


print('Logistic Auc = %2.2f'%logistic_roc_auc)

# %% [code]
