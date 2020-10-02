import re
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# outer_path >>> /kaggle/input/CORD-19-research-challenge/

# list of all of the dirs is the dataset dir
dirs = ["biorxiv_medrxiv", "comm_use_subset", "custom_license", "noncomm_use_subset"]
# dirs = ["biorxiv_medrxiv"]

docs = []
for d in dirs:
    for file in tqdm(os.listdir(f"/kaggle/input/CORD-19-research-challenge/{d}/{d}")):
        file_path = f"/kaggle/input/CORD-19-research-challenge/{d}/{d}/{file}"
        
        #load every file in the dataset into a json object
        j = json.load(open(file_path, "rb"))

        title = j['metadata']['title']
        #loading (abstract , body_text) in every json object and the exception here beacouse some files doesnot contains abstract or body_text
        try:
            body_text = j['body_text'][0]
            abstract = j['abstract']
        except:
            #storing an empty string if the exception occures
            body_text = ""
            abstract = ""

        #creating a total text of both abstract and body_text
        tot_txt = ""
        
        #appending both text of abstract and text of bodt text to the total text (tot_txt)
        for abs_text in j['abstract']:
            tot_txt += abs_text['text']

        for bod_text in j['body_text']:
            tot_txt += bod_text['text']+"\n\n"
        
        #and finally appending all of (text data) to the docs list for final preprocessing
        docs.append([title, abstract, body_text, tot_txt])

        #creating a dataframe of the docs
df = pd.DataFrame(docs, columns=['title', 'abstract', 'body_text', 'tot_txt'])

#finding each entry of ([df['tot_txt']) that contains (incubation) word
incubation = df[df['tot_txt'].str.contains("incubation")]

incubation_times = []

texts = incubation['tot_txt'].values

for t in texts:
    #preprocessing each entry to remove unwanted chars and spliting it by (. ) to turn it into sentences
    for sentence in t.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("?", "").replace(",", "").split('. '):
        #choosing only sentence that only contain (incubation) word
        if "incubation" in sentence:
            #creating regex to find only num values followed by a word day to determine that this is the (incubation value) like [( 12 day),( 4 days),() 29 day)]
            value = re.findall(r" \d{1,3} day", sentence)
            #spliting the value and appending only number values to the incubation_times
            if len(value) == 1:
                case = value[0].split(" ")
                incubation_times.append(float(case[1]))
            elif len(value) == 1:
                case_1 = value[0].split(" ")
                case_2 = value[1].split(" ")
                incubation_times.append(float(case_1[1]))
                incubation_times.append(float(case_2[1]))
            elif len(value) == 1:
                case_1 = value[0].split(" ")
                case_2 = value[1].split(" ")
                case_3 = value[2].split(" ")
                incubation_times.append(float(case_1[1]))
                incubation_times.append(float(case_2[1]))
                incubation_times.append(float(case_3[1]))

#saving all of the incubation_times data into a txt file
with open("incubation_times_mean.txt", "w") as f:
    f.write(str(incubation_times))

# with open("incubation_times_body_text.txt", "r") as f:
#     incubation_times = np.array(f.read().replace("\n", "").replace("[", "").replace("]", "").split(", "))


incubation_times = np.array(incubation_times).astype('float64')

#len of the incubatoin_times
print(f"The Len of incubation times : {len(incubation_times)}")
#mean of the incubation_time
print(f"The mean projected incubation time is {np.mean(incubation_times)} days")

#plotting the data in histogram 
plt.style.use("seaborn")
plt.hist(incubation_times, bins=50)
plt.ylabel("bins counts")
plt.xlabel("incubation times")
try:
    plt.savefig("incubation_time_histogram.jpeg")
except:
    pass
plt.show()
