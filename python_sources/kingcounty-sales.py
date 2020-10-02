import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import seaborn
		
def show_data_info(df):
    print(df.info())
    # print(df.head())
    print(df.describe())
    print(df.columns)


def load_dataset():
	dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%dT%H%M%S')
	#current_path = os.path.dirname(os.path.realpath(__file__))
	#csv_path = os.path.join(current_path, "input/kc_house_data.csv")
	return pd.read_csv('../input/kc_house_data.csv', index_col=0, parse_dates=['date'], date_parser=dateparse)


def show_histogram(df):
	
	for col in df.columns:
		plt.figure(figsize=(9, 9))
		df[col].hist(bins=50, histtype='bar', align='mid', label=col)
		plt.title(col)
		plt.legend()
		plt.grid(False)
		#print('viz/'+col+'_hist.png...saving')
		plt.savefig(col+'_hist.png', format='png', bbox_inches='tight', dpi=200)
		# plt.show()

def show_scatter(df, y_list, x_col):
	for y in y_list:
		plt.figure(figsize=(9, 9))
		plt.scatter(df[x_col], df[y])
		plt.title(y)
		plt.legend()
		plt.ylabel(y)
		plt.xlabel(x_col)
		plt.grid(False)
		#print('viz/'+y+'_scatter.png...saving')
		plt.savefig(y+'_scatter.png', format='png', bbox_inches='tight', dpi=200)
		# plt.show()

def get_correlations(df):
	print("Correlations with price:")
	corr_matrix = housing_df.corr()
	print(corr_matrix['price'].sort_values(ascending=False))


def get_zero_cols(df):
	for cols in df.columns:
		print(cols, (df[cols] == 0.0).sum())


def remove_irrelevant_colums(df_train, df_test):
	# df_train.pop('id')
	df_train.pop('date')

	# df_test.pop('id')
	df_test.pop('date')


housing_df = load_dataset()
# show_data_info(housing_df)
# show_histogram(housing_df)
get_correlations(housing_df)
# show_scatter(housing_df, ['sqft_living', 'grade', 'sqft_above', 'bathrooms', 'sqft_basement', 'bedrooms'], 'price')
# get_zero_cols(housing_df)

housing_df_train, housing_df_test = train_test_split(housing_df, test_size=0.2, random_state=42)
print('*',len(housing_df_train), ' -> ', len(housing_df_test))

remove_irrelevant_colums(housing_df_train, housing_df_test)

housing_df_train_label = housing_df_train.pop('price')
housing_df_test_label = housing_df_test.pop('price')

lin_req = LinearRegression()
lin_req.fit(housing_df_train, housing_df_train_label)

housing_df_prediction = lin_req.predict(housing_df_test)

accuracy = lin_req.score(housing_df_test, housing_df_test_label)
print("Accuracy: {}%".format(int(round(accuracy * 100))))

print(housing_df_prediction)

lin_mae = mean_absolute_error(housing_df_test_label, housing_df_prediction)
print("MAE:",lin_mae)

result_df = pd.DataFrame({"Actual": housing_df_test_label, "Predicted": housing_df_prediction})
# show_data_info(result_df)
result_df = result_df.round({'Predicted': 1})
# print(result_df)
# result_df['Predicted'] = result_df['Predicted'].astype(int)
result_df.to_csv('results.csv', header=True)

