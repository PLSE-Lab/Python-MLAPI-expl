import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

hr_comma = pd.read_csv('../input/HR_comma_sep.csv')
hr_comma.columns = ['satis_level', 'last_eva',
                    'number_project', 'ave_monthly_hours',
                    'time_spend_company', 'work_accident',
                    'left', 'promotion_last_5', 'career',
                    'salary_level']
print('columns_names:')
print(hr_comma.columns.tolist())
hr_comma['satis_level'] = hr_comma['satis_level'].astype('float')
# print(hr_comma[['satis_level', 'career']].groupby(
#     'career')['satis_level'].describe().unstack())
# print(hr_comma[['satis_level', 'work_accident']].groupby(
#     'work_accident')['satis_level'].describe().unstack())
# print(hr_comma[['satis_level', 'promotion_last_5']].groupby(
#     'promotion_last_5')['satis_level'].describe().unstack())
# print(hr_comma[['satis_level', 'left']].groupby(
#     'left')['satis_level'].describe().unstack())

'''
heatmap
'''
plt.figure(figsize=(11, 11))
sns.heatmap(hr_comma.corr(), vmax=1, square=True,
            annot=True, cmap='cubehelix',)
plt.xticks(rotation=20)
plt.yticks(rotation=20)
plt.title('Correlation between different fearures', fontsize=12)
plt.show()

'''
boxplot
'''
career_arr = []
career_labels = list(set(hr_comma['career']))
for i in career_labels:
    arrlist = []
    # arrlist.append(i)
    for j in hr_comma['satis_level'][(hr_comma['career'] == i)]:
        arrlist.append(j)
    career_arr.append(arrlist)
career_arr = np.array(career_arr)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), sharey=True)
p1 = plt.subplot(311)
p1.boxplot(
    x=([i for i in career_arr]),
    labels=career_labels,
    bootstrap=10000,
    meanline=True,
    meanprops=dict(linestyle='--', linewidth=2.5, color='purple'),
    showmeans=True,
    flierprops=dict(marker='o', markerfacecolor='green', markersize=12,
                    linestyle='none'),
    boxprops=dict(color='black'),
    # notch=True,
    patch_artist=True
)
p1.set_title('different job', fontsize=10, color='blue')

transdict = {0: 'no', 1: 'yes'}
translabel = []
work_accident_arr = []
work_accident_labels = list(set(hr_comma['work_accident']))
for i in work_accident_labels:
    arrlist = []
    # arrlist.append(i)
    for j in hr_comma['satis_level'][(hr_comma['work_accident'] == i)]:
        arrlist.append(j)
    work_accident_arr.append(arrlist)
for x in work_accident_labels:
    translabel.append(transdict[x])
work_accident_labels = translabel
work_accident_arr = np.array(work_accident_arr)
axes[1][0].boxplot(
    x=([i for i in work_accident_arr]),
    labels=work_accident_labels,
    bootstrap=10000,
    meanline=True,
    meanprops=dict(linestyle='--', linewidth=2.5, color='purple'),
    showmeans=True,
    flierprops=dict(marker='o', markerfacecolor='green', markersize=12,
                    linestyle='none'),
    boxprops=dict(color='black'),
    # notch=True,
    patch_artist=True
)
axes[1][0].set_title('work accident', fontsize=10, color='blue')

translabel = []
left_arr = []
left_labels = list(set(hr_comma['left']))
for i in left_labels:
    arrlist = []
    # arrlist.append(i)
    for j in hr_comma['satis_level'][(hr_comma['left'] == i)]:
        arrlist.append(j)
    left_arr.append(arrlist)
for x in left_labels:
    translabel.append(transdict[x])
left_labels = translabel
left_arr = np.array(left_arr)
axes[1][1].boxplot(
    x=([i for i in left_arr]),
    labels=left_labels,
    bootstrap=10000,
    meanline=True,
    meanprops=dict(linestyle='--', linewidth=2.5, color='purple'),
    showmeans=True,
    flierprops=dict(marker='o', markerfacecolor='green', markersize=12,
                    linestyle='none'),
    boxprops=dict(color='black'),
    # notch=True,
    patch_artist=True
)
axes[1][1].set_title('left', fontsize=10, color='blue')

translabel = []
promotion_last_5_arr = []
promotion_last_5_labels = list(set(hr_comma['promotion_last_5']))
for i in promotion_last_5_labels:
    arrlist = []
    # arrlist.append(i)
    for j in hr_comma['satis_level'][(hr_comma['promotion_last_5'] == i)]:
        arrlist.append(j)
    promotion_last_5_arr.append(arrlist)
for x in promotion_last_5_labels:
    translabel.append(transdict[x])
promotion_last_5_labels = translabel
promotion_last_5_arr = np.array(promotion_last_5_arr)
axes[2][0].boxplot(
    x=([i for i in promotion_last_5_arr]),
    labels=promotion_last_5_labels,
    bootstrap=10000,
    meanline=True,
    meanprops=dict(linestyle='--', linewidth=2.5, color='purple'),
    showmeans=True,
    flierprops=dict(marker='o', markerfacecolor='green', markersize=12,
                    linestyle='none'),
    boxprops=dict(color='black'),
    # notch=True,
    patch_artist=True
)
axes[2][0].set_title('promotion in last 5 years', fontsize=10, color='blue')

salary_level_arr = []
salary_level_labels = list(set(hr_comma['salary_level']))
for i in salary_level_labels:
    arrlist = []
    # arrlist.append(i)
    for j in hr_comma['satis_level'][(hr_comma['salary_level'] == i)]:
        arrlist.append(j)
    salary_level_arr.append(arrlist)
salary_level_arr = np.array(salary_level_arr)
axes[2][1].boxplot(
    x=([i for i in salary_level_arr]),
    labels=salary_level_labels,
    bootstrap=10000,
    meanline=True,
    meanprops=dict(linestyle='--', linewidth=2.5, color='purple'),
    showmeans=True,
    flierprops=dict(marker='o', markerfacecolor='green', markersize=12,
                    linestyle='none'),
    boxprops=dict(color='black'),
    # notch=True,
    patch_artist=True
)
axes[2][1].set_title('salary level', fontsize=10, color='blue')
fig.suptitle("describition of different condition", fontsize=16, color='red')
fig.subplots_adjust(hspace=0.4)
plt.show()

'''
Principal Component Analysis
'''
sales = hr_comma.groupby('career')[hr_comma.drop(
    labels=['career', 'salary_level'], axis=1).columns.tolist()].apply(sum)
hr_comma_drop = hr_comma.drop(
    labels=['career', 'salary_level'], axis=1)
cols = hr_comma_drop.columns.tolist()
cols.insert(0, cols.pop(cols.index('left')))
print(cols)
hr_comma_drop = hr_comma_drop.reindex(columns=cols)
hr_comma_drop_params = hr_comma_drop.iloc[:, 1:]
hr_comma_drop_result = hr_comma_drop.iloc[:, 0]
print(np.shape(hr_comma_drop_params))
print(np.shape(hr_comma_drop_result))

'''
standardisation
'''
hr_comma_drop_params_standar = StandardScaler(
).fit(hr_comma_drop_params)
# print(hr_comma_drop_params_standar.mean_)
hr_comma_drop_params_standar = hr_comma_drop_params_standar.transform(
    hr_comma_drop_params)

mean_vec = np.mean(hr_comma_drop_params_standar, axis=0)
print('mean_vec', mean_vec)
# mean_vec = mean_vec.repeat(hr_comma_drop_params_standar.shape[0]).reshape(
#     hr_comma_drop_params_standar.T.shape[1], hr_comma_drop_params_standar.T.shape[0])
# print(hr_comma_drop_params_standar.shape)
# print(mean_vec.shape)
cov_mat = (hr_comma_drop_params_standar - mean_vec).T.dot(hr_comma_drop_params_standar -
                                                          mean_vec) / (hr_comma_drop_params.shape[0] - 1)
# print(cov_mat)

plt.figure(figsize=(8, 8))
sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different features')
plt.show()

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('vecs\n', eig_vecs)
print('vals\n', eig_vals)
eig_sort = [(np.abs(eig_vals[i]), eig_vecs[:, i], cols[i + 1])
            for i in range(len(eig_vals))]
eig_sort.sort(key=lambda x: x[0], reverse=True)

eig_nameList = []
for i in eig_sort:
    eig_nameList.append(i[2])
print('namelist', eig_nameList)
plt.figure(figsize=(12, 6))
plt.bar(range(7), [i / sum(eig_vals) for i in sorted(eig_vals, reverse=True)], alpha=0.5, align='center',
        label='individual explained variance', width=0.35)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

matrix_x = np.hstack((eig_sort[0][1].reshape(7, 1),
                      eig_sort[1][1].reshape(7, 1),
                      eig_sort[2][1].reshape(7, 1),
                      eig_sort[3][1].reshape(7, 1),
                      eig_sort[4][1].reshape(7, 1)))
# print(matrix_x)
result = hr_comma_drop_params_standar.dot(matrix_x)
print(cols[1:6])
print('result\n',result)
