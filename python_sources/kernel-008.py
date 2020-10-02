# https://www.kaggle.com/rteja1113/lightgbm-with-latest-data
import pandas as pd
pd.read_csv("../input/sub_lgb_balanced99.csv").to_csv("submission.csv", index=False)