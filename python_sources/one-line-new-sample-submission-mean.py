import pandas as pd;pd.read_csv('../input/stage_1_sample_submission.csv', converters={'PredictionString':lambda p:'0.31 391 363 220 334'}).to_csv('sub_mean.csv', index=False)