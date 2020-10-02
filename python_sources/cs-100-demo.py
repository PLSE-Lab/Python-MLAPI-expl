import csv
import math
from pprint import pprint

CLASS_FIELD = 'Survived'

FIELD_SUMMARY_TYPES = {
    'Sex': 'bin_prob',
    'Age': 'mean_stdev',
    'Pclass': 'bin_prob',
    'Fare': 'mean_stdev',
}

def load_csv(filename):
    dicts = []
    with open(filename) as infile:
        data = csv.reader(infile)
        fields = next(data)
        for row in csv.reader(infile):
            dicts.append(dict(zip(fields, row)))
    return dicts

def partition_by(l, k):
    parts = {}
    for d in l:
        l = parts.setdefault(d[k], [])
        del d[k]
        l.append(d)
    return parts

def summarize_fields(l):
    summaries = {}
    for k, v in FIELD_SUMMARY_TYPES.items():
        if v == 'bin_prob':
            summaries[k] = {}
        else:
            summaries[k] = {'sum': 0, 'sumsq': 0, 'count': 0}
            
    for d in l:
        for k, v in FIELD_SUMMARY_TYPES.items():
            if v == 'bin_prob':
                count = summaries[k].setdefault(d[k], 0)
                summaries[k][d[k]] = count + 1
            else:
                if d[k].strip():
                    val = float(d[k])
                    summaries[k]['sum']   += val
                    summaries[k]['sumsq'] += val * val
                    summaries[k]['count'] += 1

    for k, v in FIELD_SUMMARY_TYPES.items():
        if v == 'bin_prob':
            total = sum(summaries[k].values())
            for sk, sv in summaries[k].items():
                summaries[k][sk] = sv / total
        else:
            d = summaries[k]
            s, ssq, n = d['sum'], d['sumsq'], d['count']
            del d['sum'], d['sumsq'], d['count']
            d['mean'] = s / n
            d['variance'] = (ssq - s*s/n) / n
            d['stdev'] = math.sqrt(d['variance'])
            d['pdf'] = norm_pdf(d['mean'], d['variance'])
            
    return summaries

def norm_pdf(mean, var):
    return lambda x: math.exp(-(x-mean)**2/(2*var)) / math.sqrt(2*math.pi*var)
    
if __name__ == '__main__':
    train_data = partition_by(load_csv('../input/train.csv'), CLASS_FIELD)
    summaries = {}
    for category in train_data:
        summaries[category] = summarize_fields(train_data[category])
    #pprint(summaries)

    test_data = load_csv('../input/test.csv')

    print('PassengerId,Survived') # print CSV file header

    for d in test_data:
     for k in [k1 for k1 in d if k1 not in FIELD_SUMMARY_TYPES]:
         del d[k] # prune out unneeded keys for pretty printing

    predictions = {}
    for category in train_data:
        p = 1
        for k, v in FIELD_SUMMARY_TYPES.items():
            if v == 'bin_prob':
                p *= summaries[category][k][d[k]]
            else:
                if d[k].strip():
                    p *= summaries[category][k]['pdf'](float(d[k]))
        predictions[category] = p

        #CSV result row
        print(d['PassengerId'], ',',
              '0' if predictions['0'] > predictions['1'] else '1',
              sep='')

        # pretty-printed output
        #predicted_category = 'Survived' if predictions['1'] > predictions['0'] else 'Drowned'
        #print(d, '=>', predicted_category)