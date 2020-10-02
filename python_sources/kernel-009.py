# https://www.kaggle.com/alexanderkireev/deep-learning-support-966
import pandas as pd
pd.read_csv("../input/dl_support.csv").to_csv("submission.csv", index=False)