import scipy.io
import pandas as pd

mat_file = '/kaggle/input/milling-data-set-prognostic-data/mill/mill.mat'

mat = scipy.io.loadmat(mat_file)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

# parsing arrays in arrays in mat file  
data = {}
for k,v in mat.items():
    arr = v[0]
    for i in range(len(arr)):
        sub_arr = v[0][i]
        lst= []
        for sub_index in range(len(sub_arr)):
            vals = sub_arr[sub_index][0][0]
            lst.append(vals)
        data['row_{}'.format(i)] = lst
        
data_file = pd.DataFrame.from_dict(data, orient='index', columns=['case',
                                                                  'run',
                                                                  'VB', 
                                                                  'time',
                                                                  'DOC',
                                                                  "feed", 
                                                                  "material", 
                                                                  "smcAC",
                                                                 "smcDC",
                                                                  "vib_table",
                                                                  "vib_spindle",
                                                                  "AE_table",
                                                                  "AE_spindle"])
data_file.to_csv("mill.csv")
print("DONE")

