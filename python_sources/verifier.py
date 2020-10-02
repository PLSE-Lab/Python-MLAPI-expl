import os
print(os.listdir('../input'))
import requests
import csv
requests.adapters.DEFAULT_RETRIES = 2
f = open('../input/url-list/url_list.csv', 'rt', encoding = "ISO-8859-1")
total = len([i for i in f])
f = open('../input/url-list/url_list.csv', 'rt', encoding = "ISO-8859-1")
completed, written = 0, 0
with open('../working_list.csv', 'a+', newline='') as fd:
    writer = csv.writer(fd, delimiter=',')
    for i in f:
        _ = os.system('cls')
        print('Total: ', total)
        print('Completed: %d/%d' %(completed, total))
        print('Written: ', written)
        try:
            request = requests.get('http://'+i)
            completed+=1
            if request.status_code == 200:
                writer.writerow([str('http://'+i)])
                written+=1
        except:
            pass
