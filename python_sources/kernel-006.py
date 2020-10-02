# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data
import pandas as pd
pd.read_csv("../input/sub_lgb_balanced99.csv").to_csv("submission.csv", index=False)