import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input/"))

test_files = ['../input/lewis-undersampler-9562-version/pred.csv',
              '../input/deep-learning-support/dl_support.csv',
              '../input/lightgbm-lb-0-9675/submission.csv',
              '../input/talkingdata-wordbatch-fm-ftrl/wordbatch_fm_ftrl.csv',
             '../input/xgboostp5/xgb_p3.csv',
             '../input/talkingdatalgb/sub_lgb_balanced99_p1.csv',
             '../input/talkingdatalgb/sub_lgb_balanced99_p2.csv',
             '../input/talkingdatalgb/sub_lgb_balanced99_p3.csv']


# Any results you write to the current directory are saved as output.

model_test_data = []
for test_file in test_files:
    print('read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

weights = [0.05, 0.1, 0.5, 0.1, 0.1, 0.05, 0.05, 0.05]
column_name = 'is_attributed'

print('predict')
test_predict_column = [0.] * len(model_test_data[0][column_name])
for ind in range(0, n_models):
    test_predict_column += model_test_data[ind][column_name] * weights[ind]

print('make result')
final_result = model_test_data[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {column_name: test_predict_column})), axis=1)
final_result.to_csv("average_result.csv", index=False)