# https://www.kaggle.com/sergeyzlobin/wanna-blend
import pandas as pd
pd.read_csv("../input/average_result.csv").to_csv("submission.csv", index=False)