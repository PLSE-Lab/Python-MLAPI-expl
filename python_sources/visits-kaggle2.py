import pandas
import numpy
from os.path import isfile
import math

if isfile("train.csv"):
	all_data = pandas.read_csv("train.csv")
	vis = numpy.array( [numpy.array(list(map(int, elem.split()))) for elem in all_data["visits"]] )

	wd = []
	for x in vis:
		_wd = []
		for y in x:
			_wd.append((y - 1) % 7)
		wd.append(_wd)

	save = []
	j = 0
	for x in vis:
		stat_x = [0,0,0,0,0,0,0]
		wd_id = wd[j]
		number_r = int(len(x) / 2)
		i = number_r
		while i < len(x):
			stat_x[wd_id[i]] += 1
			i += 1;
		i = 0
		save.append(stat_x)
		j += 1

	res = []
	g = 0
	for x in save:
		j = 0
		maxn = x[j]
		num = [j,]
		j += 1
		last_day = wd[g][-1]
		g += 1
		while j < len(x):
			if maxn < x[j]:
				maxn = x[j]
				num =[j,]
			else:
				if maxn == x[j]:
					num.append(j)
			j += 1

		if len(num) == 1 or last_day == 6 or last_day > num[len(num)-1]:
			r = num[0]
		else:
			i = 0
			while num[i] < last_day:
				i += 1
			if num[i] == last_day:
				if i == len(num)-1:
					r = num[0]
				else:
					r = num[i+1]
			else:
				r = num[i]
		res.append(r)

	with open('solution.csv', 'w') as file_res:
		file_res.write('id,nextvisit\n')
		i = 0;
		for x in res:
			file_res.write(str(i+1)+', '+str(res[i]+1)+'\n')
			i += 1