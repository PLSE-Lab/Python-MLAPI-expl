import numpy as np
import pandas as pd

data = pd.read_csv('../input/train.csv')

def get_user_matrix(data):
    cnt = 1
    n = data.shape[0]
    ret = {}
    for i in range(data.shape[0]):
        tmp_user = np.zeros(1099, dtype = np.int8)
        tmp_user[np.array(list(map(int, data.iloc[i, 1].split(' ')[1:])))-1]= 1
        print(len(tmp_user)/7)
        tmp_user = tmp_user.reshape(int(len(tmp_user)/7), 7)
        ret[data.iloc[i, 0]] = tmp_user
        if cnt % 3000 == 0:
            print(cnt/float(n))
        cnt += 1
    return ret

users_dict = get_user_matrix(data)
no_zero_week_dict = {key:users_dict[key][users_dict[key].sum(axis=1) > 0, :]
for key in users_dict.keys()}

def prob(mat, delta):
    w = np.arange(len(mat)) + 1
    w = 1 / (w ** delta)
    w = w[::-1]
    w = w / w.sum()
    mat = mat * w.reshape((len(w), 1))
    return mat.sum(axis=0)

final_func = lambda x: prob(x, 0.2)

probs_dict = {key:final_func(no_zero_week_dict[key]) for key in no_zero_week_dict.keys()}
probs = pd.DataFrame.from_dict(probs_dict, orient='index').values
first_probs = np.zeros((probs.shape[0], 7))
first_probs[:, 0] = probs[:, 0]
for i in range(6):
    first_probs[:, i+1] = (1 - probs)[:, :i+1].prod(axis=1) * probs[:, i+1]
	
result = pd.DataFrame({'id': np.arange(first_probs.shape[0])+1, 'nextvisit':pd.Series(first_probs.argmax(axis=1) + 1).apply(lambda x: ' ' + str(x))})
result.to_csv('result.csv', index=False)	