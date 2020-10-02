# https://www.kaggle.com/prashantkikani/talkingdata-simple-blend
import pandas as pd
pd.read_csv("../input/blend_2.csv").to_csv("submission.csv", index=False)