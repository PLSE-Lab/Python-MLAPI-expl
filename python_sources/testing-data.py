import numpy as np
import matplotlib.pyplot as plt

def parse_int(i):
    try:
        return int(i)
    except:
        return 666


def _get_col_names(path):
    with open(path, 'r') as fh:
        for line in fh:
            row = line.rstrip('\n').split(",")
            return row


def _get_data(path, indexes):
    list_data=[]
    with open(path, 'r') as fh:
        next(fh)  #skip header
        for line in fh:
            row = line.rstrip('\n').split(",")
            r_temp =[parse_int(row[i]) for i in indexes]
            list_data.append(r_temp)
    return list_data


def extract_data(d_type, part, cnames):
    if d_type == 'house':
        path = '../input/pums//ss13pus'
    elif d_type == 'person':
        path = '../input/pums//ss13hus'
    else:
        print ('type can be house or person type data')
        return True
    
    if part == 'a':
        path = [path + 'a.csv']
    elif part == 'b':
        path = [path + 'b.csv']
    elif part == 'both':
        path = [path + 'a.csv', path + 'b.csv']
    else:
        print ('type a, b or both')
        
    header = _get_col_names(path[0])
    num_col_index = [header.index(col) for col in cnames]
    col_names = ({cnames[i]:i for i in range(len(cnames))})
    
    darray = np.empty([0,len(cnames)])
    for p in path:
        list_data = _get_data('../input/pums//ss13pusa.csv', num_col_index)
        array_temp = np.array(list_data)
        darray = np.vstack([darray, array_temp])
    return darray, col_names


######################################################################################

cols_needed_numerical = ['SSP', 'WAGP', 'ANC2P']    # define which columns we need
darray, index_names =  extract_data('house', 'both', cols_needed_numerical)  # exctract data where darray is array with data and index_names is dict with column names

plt.hist(darray[:, 0], bins=50) # draw hist
plt.savefig("hist_output.png")  # save the hist

