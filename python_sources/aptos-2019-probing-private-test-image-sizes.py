#!/usr/bin/env python
# coding: utf-8

# # Does the private test set have the same image size distribution as the public test set?
# According to [this really cool kernel](https://www.kaggle.com/fhopfmueller/removing-unwanted-correlations-in-training-public), 72.770% of the public test images has the size of 640x480. Is this ratio the same in the private test set? The answer to this question would give us some clues about how much difference there is between the public test set and the private test set. In this kernel, inspired by [this discussion](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/97652#latest-596913) and [this kernel](https://www.kaggle.com/cdeotte/private-lb-probing-0-950), the image size distribution of public + private test set will be probed.
# 
# # Approach
# Let's assume that we have N-(submission.csv, public LB score) pairs from already submitted kernels. By changing the submission file according to the desired information about the private test set, we can get log(N) bits information from one submission. Here "change submission" means to copy the targets of the public test set from the known submissions and insert arbitrary dummy targets to the remaining private test set. By doing so, we can control the public LB score according to what we want to know.
# 
# # Answer
# The answer is no. While the ratio of 640x480 images in the public test set is 72.770%, the ratio of 640x480 images in the public + private test set is 30-40%. There are some differences between the public test set and the private test set. I hope this information would be useful in choosing the final submission(s).

# In[ ]:


from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2


# In[ ]:


# private + public test
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
df = pd.read_csv(test_csv_path)


# Let's count the number of images with the size of 640x480, and calculate the ratio of them. `target_ratio_code 0` indicates 0-10% ratio, `target_ratio_code 1` indicates 10-20% ratio, and so forth. In case of the public test set, as the ratio is 0.72770, `target_ratio_code` is 7.

# In[ ]:


test_image_dir = Path("../input/aptos2019-blindness-detection/test_images")
private_img_cnt = 0
target_img_cnt = 0

for _, row in df.iterrows():
    id_code = row["id_code"]
    img_path = test_image_dir.joinpath(f"{id_code}.png")
    img = cv2.imread(str(img_path), 1)
    h, w, _ = img.shape

    if w == 640 and h == 480:
        target_img_cnt += 1
        
    private_img_cnt += 1
    
target_ratio = target_img_cnt / private_img_cnt
target_ratio_code = int(target_ratio * 10)
print(target_ratio)
print(target_ratio_code)


# I have ten submissions with known public LB scores. Let's select a submission file according to `target_ratio_code` and copy the targets to the submission of this kernel.

# In[ ]:


submissions = [
    "submission_0.683.csv",
    "submission_0.694.csv",
    "submission_0.709.csv",
    "submission_0.711.csv",
    "submission_0.739.csv",
    "submission_0.751.csv",
    "submission_0.755.csv",
    "submission_0.766.csv",
    "submission_0.768.csv",
    "submission_0.785.csv"
]

# select a submission file according to the ratio of target image size count.
pub_test_csv_path = "../input/aptos10submissions/" + submissions[target_ratio_code]
pub_df = pd.read_csv(pub_test_csv_path)
id_to_diagnosis = {id_code: diag for id_code, diag in zip(pub_df.id_code, pub_df.diagnosis)}


# In[ ]:


all_diagnosis = []

for _, row in df.iterrows():
    id_code = row["id_code"]
    
    if id_code in id_to_diagnosis:
        all_diagnosis.append(id_to_diagnosis[id_code])
    else:
        all_diagnosis.append(0)


# In[ ]:


id_codes = df.id_code.values
new_df = pd.DataFrame.from_dict(data={"id_code": df.id_code.values, "diagnosis": all_diagnosis})
new_df.to_csv("submission.csv", index=False)


# Alright, what is the public LB score? I got the score of 0.711, which indicates that `target_ratio_code` of the public + private test set is 3; the ratio is 30-40%. This is a little bit different from that of the public test set. How do you think this?
