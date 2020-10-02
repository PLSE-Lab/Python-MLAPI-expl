import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

submissions = pd.read_csv('../input/Submissions.csv', index_col='Id', parse_dates=[2])

submissions['dayofweek'] = submissions['DateSubmitted'].dt.dayofweek
submissions['hour'] = submissions['DateSubmitted'].dt.hour
submissions['minute'] = submissions['DateSubmitted'].dt.minute
submissions['time'] = submissions['DateSubmitted'].dt.time
submissions['dayminutes'] = submissions['hour'].apply(lambda x: x*60) + submissions['minute']


plt.figure()
submissions['dayofweek'].value_counts().sort_index().plot(kind='bar')
plt.title('Submissions by Day of Week', fontsize=16)
plt.xlabel('We can see that most contests end on Monday!')
plt.savefig('by_day_of_week.png')
plt.show()

plt.figure()
submissions['hour'].value_counts().sort_index().plot(kind='bar')
plt.title('Submissions by Hour', fontsize=16)
plt.savefig('by_hour.png')
plt.show()

early_mask = submissions['hour'] == 23
early_minutes = submissions.loc[early_mask, 'minute'] - 60
late_mask = submissions['hour'] == 0
late_minutes = submissions.loc[late_mask, 'minute']

plt.figure()
pd.concat([early_minutes, late_minutes], axis=0).hist(bins=np.arange(-60,61),
          normed=1, histtype='stepfilled')
plt.title('Submissions by Minute (around 0 GMT)', fontsize=16)
plt.xlabel('People watch the clock, waiting for submission count to reset!')
plt.xlim((-60,60))
plt.savefig('by_minute.png')
plt.show()