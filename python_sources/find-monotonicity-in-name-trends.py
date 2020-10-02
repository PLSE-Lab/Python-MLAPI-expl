"""
Find the names that are monotonically changing over time

Experiment with the min_year and max_year to see that there are indeed intervals of time in
which names come into of go out of favor.

Interestingly, there is no name in the data set that grows or decreases over the whole century.

"""
import pandas as pd

csv_file_name = '../input/NationalNames.csv'
df = pd.read_csv(csv_file_name, sep=',', header=0)


#min_year = df['Year'].min()
#max_year = df['Year'].max()
min_year=1920
max_year=2012

d = {}
for y in range(min_year, max_year - min_year + min_year + 1):
    df_for_year = df[['Name', 'Gender', 'Count']][df['Year'] == y]
    col_names = ['Name', 'Gender', 'Count'+str(y)]
    df_for_year.columns = col_names
    d[y] = df_for_year 

incr_df = d[min_year]
decr_df = d[min_year]
print('incr_df rows before: {}'.format(incr_df.shape[0]))
print('decr_df rows before: {}'.format(decr_df.shape[0]))
old_y = min_year
for y in range(min_year + 1, max_year - min_year + min_year + 1):
    incr_df = pd.merge(incr_df, d[y], how='inner', on=['Name', 'Gender'])
    incr_df = incr_df[incr_df['Count'+str(y)] > incr_df['Count'+str(old_y)]]
    decr_df = pd.merge(decr_df, d[y], how='inner', on=['Name', 'Gender'])
    decr_df = decr_df[decr_df['Count'+str(y)] < decr_df['Count'+str(old_y)]]
    old_y = y
print('incr_df rows after: {}'.format(incr_df.shape[0]))
print('decr_df rows after: {}'.format(decr_df.shape[0]))
