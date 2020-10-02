import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
submission = pd.read_csv('../input/sample_submission.csv')
submission.to_csv('submission.csv', index=False)
