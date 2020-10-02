import numpy as np
import pandas as pd

submission = pd.read_csv('../input/gender_submission.csv')
submission['Survived'] = np.round(np.random.random((len(submission)))).astype(int)
submission.to_csv('vintage_random_submission.csv', index=False)