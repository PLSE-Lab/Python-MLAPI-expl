# https://www.kaggle.com/aharless/lightgbm-smaller
import pandas as pd
pd.read_csv("../input/submission.csv").to_csv("submission.csv", index=False)