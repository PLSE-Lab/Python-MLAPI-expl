with open('../input/train.csv', 'r') as f:
    train_values = []
    for line in f.readlines():
        train_values.append(line.split(',')[1].lstrip())

answer = ['id,nextvisit']
for row_idx, row in enumerate(train_values):
    if row_idx:
        for_day, arr = [0, 0, 0, 0, 0, 0, 0], []
        for el in row.split():
            arr.append(int(el))
        if arr[-1] >= 1010:
            for idx in arr:
                for_day[(idx - 1) % 7] += idx
            max_idx, max_el = 0, for_day[0]
            for idx, el in enumerate(for_day):
                if el > max_el:
                    max_idx, max_el = idx, el
            answer.append(str(row_idx)+', '+str(max_idx + 1))
        else:
            answer.append(str(row_idx)+', 0')

with open('./solution.csv', 'w') as f:
    f.write('\n'.join(answer))