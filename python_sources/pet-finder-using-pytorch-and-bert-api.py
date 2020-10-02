#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install  "../input/pytorchbert/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master"')


# In[ ]:


enable_bert = 1
if enable_bert==2:
    get_ipython().system('pip install pytorch-pretrained-bert==0.4.0')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch, torchvision
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
import sklearn.model_selection as skms
if enable_bert==1:
    from pytorch_pretrained_bert import BertTokenizer
    from pytorch_pretrained_bert.modeling import BertModel
print(os.listdir("../input"))


# In[ ]:


enable_bert


# In[ ]:


img_size = 224
n_image_features = 5


# In[ ]:


class image_dataset():
    def __init__(self, path, dic_PetID, list_PetIDSNo):
        self.img_list = os.listdir(path)
        #self.img_list = self.img_list[0:500]
        self.train_image_path = path
        self.transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(),
                                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.50,0.5))])
        ds = pd.DataFrame({"filename":self.img_list})
        ds["PetID"] = ds["filename"].map(lambda x: x.split("-")[0])
        ds["PetID_SNo"] = ds["PetID"].map(dic_PetID) 
        ds = ds[ds["PetID_SNo"].isin(list_PetIDSNo)]
        self.PetID_SNo = list(ds["PetID_SNo"])
    def __getitem__(self, index):
        img_file = self.img_list[index]
        img = Image.open(self.train_image_path + img_file).convert('RGB') 
        img1 = self.transform(img)
        label = torch.tensor(self.PetID_SNo[index])
        return img1, label
    def __len__(self):
        return len(self.PetID_SNo)   


# In[ ]:


#The output of this step is representation of data
def data_collection():
    breed_labels = pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
    color_labels = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
    state_labels = pd.read_csv("../input/petfinder-adoption-prediction/state_labels.csv")
    train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    sample_submission = pd.read_csv("../input/petfinder-adoption-prediction/test/sample_submission.csv")
    
    dic_PetID = {val:idx for idx,val in enumerate(list(set(train["PetID"].unique()) | set(test["PetID"].unique())))}
    train["PetID_SNo"] = train["PetID"].map(dic_PetID) 
    test["PetID_SNo"] = test["PetID"].map(dic_PetID) 

    return breed_labels, color_labels, state_labels, train, test, sample_submission, dic_PetID


# In[ ]:


breed_labels, color_labels, state_labels, train, test, sample_submission, dic_PetID = data_collection()


# In[ ]:


color_labels.head(10)


# In[ ]:


train.info()


# In[ ]:


train.head(1)


# In[ ]:


train.describe()


# In[ ]:


for col in train.columns:
    print(col,len(train[col].unique()))


# In[ ]:


def get_dummy_col(colName, blndummy=True):
    global train, test
    if blndummy==True:
        df_grouped = train.groupby([colName], as_index=False).agg({"AdoptionSpeed":["mean"]})
        df_grouped.columns =[colName,"AdoptionSpeed"]
        df_grouped.plot(x=colName,y="AdoptionSpeed", figsize=(6,3), kind="barh")

        df_type = pd.get_dummies(train[colName], drop_first=False)
        df_type.columns = [colName + str(col) for col in df_type.columns]
        train.drop(colName, inplace=True, axis=1)
        train = pd.concat([train,df_type], axis=1)

        df_type = pd.get_dummies(test[colName], drop_first=False)
        df_type.columns = [colName + str(col) for col in df_type.columns]
        test.drop(colName, inplace=True, axis=1)
        test = pd.concat([test,df_type], axis=1)
    else:
        train.drop(colName, inplace=True, axis=1)
        test.drop(colName, inplace=True, axis=1)


# In[ ]:


get_dummy_col("Type")


# In[ ]:


train.drop("Name",inplace=True, axis=1)
test.drop("Name",inplace=True, axis=1)


# In[ ]:


train["Age"].hist(bins=200, figsize=(12,4))


# In[ ]:


if 1 == 2:
    df_test_train = pd.concat((train[["Breed1","Breed2","Age"]],test[["Breed1","Breed2","Age"]]), axis=0 )
    df_grouped = df_test_train.groupby(["Breed1","Breed2"], as_index=False).agg({"Age":["max"]})
    df_grouped.columns =["Breed1","Breed2","MaxAge"]
    for df in [train,test]:
        df_merged = pd.merge(df.reset_index()[["index","Breed1","Breed2"]], df_grouped, how="inner", on = ["Breed1","Breed2"]).set_index("index")
        df["MaxAge"] = 0
        df.loc[df_merged.index, "MaxAge"] = df_merged.loc[df_merged.index, "MaxAge"]
        df["NormalizedAge"] = df["Age"]/df["MaxAge"]
        df["NormalizedAge"] = (df["NormalizedAge"]*30).round(0).fillna(0)
        df.drop(["MaxAge","Age"],inplace=True, axis=1)
    get_dummy_col("NormalizedAge")


# In[ ]:





# In[ ]:


for df in [train,test]:
    pd_merge = pd.merge(df.reset_index(), breed_labels, how="left", left_on="Breed1", right_on="BreedID").set_index("index")
    df.loc[pd_merge.index, "Breed1Type"] = pd_merge.loc[pd_merge.index, "Type"]
    df.loc[pd_merge.index, "Breed1Name"] = pd_merge.loc[pd_merge.index, "BreedName"]
    df.drop("Breed1", inplace=True, axis=1)
get_dummy_col("Breed1Type")


# In[ ]:





# In[ ]:


get_dummy_col("Breed1Name",False)


# In[ ]:


for df in [train,test]:
    pd_merge = pd.merge(df.reset_index(), breed_labels, how="left", left_on="Breed2", right_on="BreedID").set_index("index")
    df.loc[pd_merge.index, "Breed2Type"] = pd_merge.loc[pd_merge.index, "Type"]
    df.loc[pd_merge.index, "Breed2Name"] = pd_merge.loc[pd_merge.index, "BreedName"]
    df.drop("Breed2", inplace=True, axis=1)
get_dummy_col("Breed2Type")


# In[ ]:


get_dummy_col("Breed2Name",False)


# In[ ]:


get_dummy_col("Gender")


# In[ ]:


get_dummy_col("Color1")


# In[ ]:


get_dummy_col("Color2")


# In[ ]:


get_dummy_col("Color3")


# In[ ]:


get_dummy_col("MaturitySize")


# In[ ]:


get_dummy_col("FurLength")


# In[ ]:


get_dummy_col("Vaccinated")


# In[ ]:


get_dummy_col("Dewormed")


# In[ ]:


get_dummy_col("Sterilized")


# In[ ]:


get_dummy_col("Health")


# In[ ]:


get_dummy_col("State")


# In[ ]:


train.drop("RescuerID", axis=1, inplace=True)
test.drop("RescuerID", axis=1, inplace=True)
#train.drop("Description", axis=1, inplace=True) #For now we are dropping description but we will use it later
train.head(10)


# In[ ]:


train_feature_cols = list(train.columns.values)
train_feature_cols.remove("PetID")
train_feature_cols.remove("PetID_SNo")
train_feature_cols.remove("AdoptionSpeed")
train_feature_cols.remove("Description")


# In[ ]:


test_feature_cols = list(test.columns.values)
test_feature_cols.remove("PetID")
test_feature_cols.remove("PetID_SNo")
test_feature_cols.remove("Description")


# In[ ]:


for col in list(set(train_feature_cols) - set(test_feature_cols)):
    test[col] = 0


# In[ ]:


for col in train_feature_cols:
    if train[train[col].isna()].shape[0] > 0:
        print(col)


# In[ ]:


class PetModel(nn.Module):
    def __init__(self, n_input, lst_hidden, n_output):
        super().__init__()
        input_features = n_input
        self.hidden_layers = []
        for hidden in lst_hidden:
            output_features = hidden
            layer = nn.Linear(input_features, output_features)
            self.hidden_layers.append(layer)
            input_features = hidden
        self.last_layer = nn.Linear(input_features, n_output)
    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.last_layer(x)
        return x
        


# In[ ]:


lst_hidden = [48,32,16]
model = PetModel(len(train_feature_cols), lst_hidden, 5)
#10 -> 1.52
#[32,16]1.465


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 200


# In[ ]:


x_all = train[train_feature_cols].values.astype(np.float32)
x_test = test[train_feature_cols].values.astype(np.float32)
y_all = train["AdoptionSpeed"].values
PetID_SNo_all = train["PetID_SNo"].values
PetID_SNo_test = test["PetID_SNo"].values
x_train, x_val, y_train, y_val, PetID_SNo_train, PetID_SNo_val = skms.train_test_split(x_all, y_all,PetID_SNo_all, stratify=y_all)
x_train, x_val, y_train, y_val, x_test = torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(x_test)


# In[ ]:


train_losses = []
val_losses = []
trend = []
for i in range(epochs):
    pred = model.forward(x_train)
    optimizer.zero_grad()
    loss = criterion(pred,y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    
    pred_val = model.forward(x_val)
    val_loss = criterion(pred_val,y_val)
    val_losses.append(val_loss.item())
    pred_val1 = torch.max(pred_val, 1)[1]
    acc = (torch.sum(pred_val1 == y_val)*100.0) / y_val.shape[0]
    if i %10 == 0:
        print(i, loss.item(), val_loss.item(),  loss.item()>val_loss.item(), acc)
    trend.append(loss.item()>val_loss.item())
   
    
pred_test = model.forward(x_test)
df = pd.DataFrame({"PetID":list(test["PetID"].values), "AdoptionSpeed": list(torch.max(pred_test, 1)[1].numpy())})
df.to_csv('submission.csv', index=False)


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(range(epochs), train_losses)
plt.plot(range(epochs), val_losses)


# In[ ]:



train_img_ds = image_dataset("../input/petfinder-adoption-prediction/train_images/",dic_PetID, PetID_SNo_train)
val_img_ds = image_dataset("../input/petfinder-adoption-prediction/train_images/",dic_PetID, PetID_SNo_val)
test_img_ds = image_dataset("../input/petfinder-adoption-prediction/test_images/",dic_PetID, PetID_SNo_test)
#test_img_ds = image_dataset("../input/test_images/", dic_PetID)
train_img_loader = torch.utils.data.DataLoader(dataset=train_img_ds, batch_size=20, shuffle=True)
val_img_loader = torch.utils.data.DataLoader(dataset=val_img_ds, batch_size=200)
test_img_loader = torch.utils.data.DataLoader(dataset=test_img_ds, batch_size=200)
#test_img_loader = torch.utils.data.DataLoader(dataset=test_img_ds, batch_size=500, shuffle=True)


# In[ ]:


vgg = models.vgg16() #pretrained=True)
vgg.load_state_dict(torch.load("../input/vggweights/vgg16.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)
device


# In[ ]:


for param in vgg.features.parameters():
    param.requires_grad = False


# In[ ]:


n_inputs = vgg.classifier[6].in_features
last_layer = nn.Linear(n_inputs, n_image_features)
vgg.classifier[6] = last_layer
print(vgg.classifier[6].out_features)
epochs = 10


# In[ ]:



if enable_bert == 1:
    n_text_features = 5
    bert = BertModel.from_pretrained("../input/pretrainedbert/bert-base-uncased")
    for param in bert.parameters():
        param.requires_grad = False
    bert_dropout = torch.nn.Dropout(0.1)
    bert_classifier = torch.nn.Linear(768, n_text_features)
    tokenizer = BertTokenizer.from_pretrained('../input/berttoken1/bert-base-uncased-vocab.txt', do_lower_case=True)
    train["Description"] = train["Description"].fillna("")
    test["Description"] = test["Description"].fillna("")
    MAX_LEN=100
    from keras.preprocessing.sequence import pad_sequences
else:
    n_text_features = 0


# In[ ]:


lst_hidden = [48,32,16]
model_with_image = PetModel(len(train_feature_cols)+n_image_features + n_text_features, lst_hidden, 5)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_with_image.parameters(), lr = 0.005)
epochs = 0


# In[ ]:


enable_bert


# In[ ]:


def get_loss(data, label, for_val=0, for_test=0):
    global vgg, device, train, test, tokenizer, optimizer, criterion
    if enable_bert == 1: 
        global bert, bert_dropout, bert_classifier
    data = data.to(device)
    label = label.to(device)
    pred_img1 = vgg.forward(data)
    pred_img = torch.softmax(pred_img1, dim=1)
    list_PetID_SNo = label.cpu().numpy()
    df_batch = pd.DataFrame({"PetID_SNo":list_PetID_SNo})
    if for_test == 0:
        df_merged = pd.merge(df_batch, train, how="inner",on="PetID_SNo")
    else:
        df_merged = pd.merge(df_batch, test, how="inner",on="PetID_SNo")

    if enable_bert == 1:
        sentences = df_merged["Description"].values
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
        _, pooled_output = bert(torch.from_numpy(np.array(input_ids)).to(device), None, torch.from_numpy(np.array(attention_masks)).to(device), output_all_encoded_layers=False)
        pooled_output = bert_dropout(pooled_output)
        pred_text1 = bert_classifier(pooled_output)
        pred_text = torch.softmax(pred_text1, dim=1)

    x_batch = df_merged[train_feature_cols].values.astype(np.float32)
    if enable_bert == 1:
        data_batch = torch.cat((torch.from_numpy(x_batch),torch.from_numpy(pred_img.cpu().detach().numpy().astype(np.float32)),                             torch.from_numpy(pred_text.cpu().detach().numpy().astype(np.float32))), dim=1)
    else:
        data_batch = torch.cat((torch.from_numpy(x_batch),torch.from_numpy(pred_img.cpu().detach().numpy().astype(np.float32)),                             ), dim=1)
    if for_test==0:
        label_batch = df_merged["AdoptionSpeed"].values
    
    pred = model_with_image.forward(data_batch)
    if for_val==0:
        optimizer.zero_grad()
    if for_test==0:
        loss = criterion(pred,torch.from_numpy(label_batch))
    
    if for_val==0:
        loss.backward()
        optimizer.step()
    if for_test==0:
        y_batch = torch.from_numpy(label_batch)
        pred_batch = torch.max(pred, 1)[1]
        num_correct = torch.sum(y_batch==pred_batch)
        return loss.item(), num_correct, pred
    else:
        return pd.DataFrame({"PetID":list(df_merged["PetID"].values), "AdoptionSpeed": list(torch.max(pred, 1)[1].numpy())})


# In[ ]:


vgg = vgg.to(device)
if enable_bert==1:
    bert = bert.to(device)
    bert_dropout = bert_dropout.to(device)
    bert_classifier = bert_classifier.to(device)


train_losses = []
val_losses = []
trend = []
df_final_list = []
for i in range(epochs):
    tot_loss = 0
    tot_val_loss = 0
    tot_num_correct= 0
    tot_val_num_correct = 0
    j = 0
    for data, label in train_img_loader:
        batch_loss,num_correct,_ = get_loss(data, label) 
        tot_loss += batch_loss
        tot_num_correct += num_correct
        if j % 50 == 0:
            print("epoch", i, " batch_loss", batch_loss)
        j += 1
    train_losses.append(tot_loss/(len(train_img_loader)))
    print("epoch", i, " tot_loss",tot_loss/(len(train_img_loader)),"acc", (tot_num_correct*100.00)/(len(train_img_ds)), tot_num_correct,(len(train_img_ds)))
    j = 0
    for data, label in val_img_loader:
        batch_loss,num_correct, _ = get_loss(data, label,1) 
        tot_val_loss += batch_loss
        tot_val_num_correct += num_correct
        if j % 10 == 0:
            print("epoch", i, " batch val loss",batch_loss)
        j+=1
    val_losses.append(tot_val_loss/(len(val_img_loader)))
    print("epoch", i, " val_losses",tot_val_loss/(len(val_img_loader)), "acc", (tot_val_num_correct*100.00)/(len(val_img_ds)), tot_val_num_correct, (len(val_img_ds)))
    df_list = []
    if (i==6):
        for data, label in test_img_loader:
            result_df = get_loss(data, label,1,1) 
            df_list.append(result_df)
        df_final = pd.concat(df_list)
        df_final_grouped = df_final.groupby("PetID", as_index=False).median()
        df_final_grouped.columns = list(df_final.columns.values)

        id_without_img = list(set(test["PetID"].values) - set(df_final_grouped["PetID"].values))
        filter_df = test[test["PetID"].isin(id_without_img)]
        test_x_val = filter_df[train_feature_cols].values.astype(np.float32)
        pred = model.forward(torch.from_numpy(test_x_val))
        filter_df = pd.DataFrame({"PetID":list(filter_df["PetID"].values), "AdoptionSpeed": list(torch.max(pred, 1)[1].numpy())})
        df_final_grouped = pd.concat([df_final_grouped, filter_df])
        df_final_grouped["AdoptionSpeed"] = df_final_grouped["AdoptionSpeed"].astype(np.int32)
        df_final_grouped.to_csv('submission.csv', index=False)
        df_final_list.append(df_final_grouped)


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(range(epochs), train_losses)
plt.plot(range(epochs), val_losses)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




