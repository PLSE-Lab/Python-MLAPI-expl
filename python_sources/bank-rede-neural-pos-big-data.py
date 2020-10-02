import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, precision_recall_fscore_support

dataset = pd.read_csv('../input/bank-full.csv')

# preprocessa valores numericos
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
label_binarizer = preprocessing.LabelBinarizer()

dataset['age'] = min_max_scaler.fit_transform(dataset['age'].values.reshape(-1, 1))
dataset['balance'] = min_max_scaler.fit_transform(dataset['balance'].values.reshape(-1, 1))
dataset['day'] = min_max_scaler.fit_transform(dataset['day'].values.reshape(-1, 1))
dataset['duration'] = min_max_scaler.fit_transform(dataset['duration'].values.reshape(-1, 1))
dataset['campaign'] = min_max_scaler.fit_transform(dataset['campaign'].values.reshape(-1, 1))
dataset['pdays'] = min_max_scaler.fit_transform(dataset['pdays'].values.reshape(-1, 1))
dataset['previous'] = min_max_scaler.fit_transform(dataset['previous'].values.reshape(-1, 1))

# binario para 1 ou 0
dataset['default'] = label_binarizer.fit_transform(dataset['default'].values)
dataset['housing'] = label_binarizer.fit_transform(dataset['housing'].values)
dataset['loan'] = label_binarizer.fit_transform(dataset['loan'].values)

# transforma as categorias em várias colunas
job_columns = dataset['job'].str.get_dummies().add_prefix('job_')
marital_columns = dataset['marital'].str.get_dummies().add_prefix('marital_')
education_columns = dataset['education'].str.get_dummies().add_prefix('education_')
contact_columns = dataset['contact'].str.get_dummies().add_prefix('contact_')
month_columns = dataset['month'].str.get_dummies().add_prefix('month_')
poutcome_columns = dataset['poutcome'].str.get_dummies().add_prefix('poutcome_')
y_columns = dataset['y'].str.get_dummies().add_prefix('y_')

# remove as colunas que fizemos o processamento
dataset = dataset.drop(columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'y'])

# adiciona os dados anteriores em um dataset final
final_dataset = dataset.join(job_columns)
final_dataset = final_dataset.join(marital_columns)
final_dataset = final_dataset.join(education_columns)
final_dataset = final_dataset.join(contact_columns)
final_dataset = final_dataset.join(month_columns)
final_dataset = final_dataset.join(poutcome_columns)
final_dataset = final_dataset.join(y_columns)

# salva em csv
final_dataset.to_csv('bank_preprocessed.csv', index=False)

# divide o dataset entre X e y
y = final_dataset[['y_no', 'y_yes']]
X = final_dataset.drop(columns=['y_no', 'y_yes'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

#clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = MLPClassifier(activation ='tanh', solver='lbfgs', hidden_layer_sizes=(3, 3), random_state=1, verbose=True)
#clf = BernoulliRBM(n_components=2)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(str(accuracy_score(y_test, y_pred) * 100) + '%')
print(classification_report(y_test, y_pred))
print(precision_recall_fscore_support(y_test, y_pred))