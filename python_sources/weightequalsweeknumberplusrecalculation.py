import numpy as np 
import pandas as pd 

data = pd.read_csv('../input/train.csv')

def func(x):
    return [int(x) for x in x.split()]
data.visits = data.visits.apply(func)

# simplest weight scheme 
# weight = week number

# p_vector consists of 7 values, because "0 - test" showed that data doesn't have 0 in it
# that way we'll have one class less
def calc_week(day_num):
    return day_num // 7 + 1

def calc_day(day_num):
    day = day_num % 7
    if (day == 0):
        return 7
    else:
        return day

def recalc(p_vec, day_type):
    product = 1
    for i in range(0, day_type - 1):
        product *= 1 - p_vec[i]
    product *= p_vec[day_type-1]
    return product

week_num_sum = 0
for i in range(1, 158):
    week_num_sum += i
    
p_vectors = []
for i in data.visits:
    p_vector = [0, 0, 0 ,0 ,0 ,0 ,0]
    for j in i:
        day_type = calc_day(j)
        week_num = calc_week(j)
        
        p_vector[day_type - 1] += week_num / week_num_sum
    p_vectors.append(p_vector)

#recalculating
answers = []

for item in p_vectors:
    answer_vector = [0, 0, 0 ,0 ,0 ,0 ,0]
    for i in range(1, 8):
        answer_vector[i-1] = recalc(item, i)
    answers.append(answer_vector)

# storing result
y = []

for item in answers:
    y.append(np.argmax(np.array(item)) + 1)

with open('out.csv','w') as file:
        file.write('id,nextvisit\n')
        for i in range(0, len(y)):
            file.write(str(i + 1) + ', ' + str(y[i]) + '\n')