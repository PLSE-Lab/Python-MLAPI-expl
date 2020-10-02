import numpy as np 
import pandas as pd 

# read the file
df = pd.read_csv('/kaggle/input/scl-dummy/Dummy data.csv')

# the solution dataframe
solution_df = pd.DataFrame(columns=['id','new_number'])

# plus 2 to new_number
for i in df['id']:
    solution_df = solution_df.append({'id':i,'new_number':i+2},ignore_index=True)

# information about solution dataframe
print(solution_df.head)
print(solution_df.shape)
    