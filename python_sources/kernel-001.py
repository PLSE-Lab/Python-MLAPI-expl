# https://www.kaggle.com/tunguz/psst-wanna-blend-some-more
import pandas as pd
pd.read_csv("../input/average_result.csv").to_csv("submission.csv", index=False)