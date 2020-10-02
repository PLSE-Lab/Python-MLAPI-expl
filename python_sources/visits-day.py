from pandas import read_csv, DataFrame
from numpy import float64
from sklearn.utils.extmath import weighted_mode


train_input = read_csv('../input/train.csv')
index  = [float64((i - 1) % 7) for i in range(0, 1100)]
powers = [float64(pow((i + 6) // 7, 2)) for i in range(0, 1100)]
string_represenation = [' ' + str(res + 1) for res in range(7)]

solution = DataFrame(columns = ['id', 'nextvisit'])
solution['id'] = train_input['id']
solution['nextvisit'] = train_input['visits'].apply(lambda x: string_represenation[int(
    weighted_mode([index[i] for i in [int(s) for s in x.split(' ')[1:]]],
        [powers[i] for i in [int(s) for s in x.split(' ')[1:]]])[0][0])])
solution.to_csv('solution.csv', index=False, sep =',')
