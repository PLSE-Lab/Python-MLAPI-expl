from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "../input/"
sz = 224
arch = resnet34
bs = 58

label_csv = f'{PATH}/train.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)

print(n)
print(len(val_idxs))

label_df = pd.read_csv('../input/train.csv')
print(label_df.head())

label_df.pivot_table(index="Id", aggfunc=len)

def get_data(sz, bs): # sz: image size, bs: batch size
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}train.csv', test_name='test',
                                       val_idxs=val_idxs, suffix='', tfms=tfms, bs=bs)
    return data if sz > 300 else data.resize(340, '../working')
    
data = get_data(sz, bs)
learn = ConvLearner.pretrained(arch, data, precompute=True)

#lr = learn.lr_find()
learn.fit(1e-2,2)


temp = learn.predict(is_test=True)
print(temp.shape)
pred_test = np.argmax(temp, axis=1)
print(pred_test[:20])


multi_preds, y = learn.TTA(is_test=True)
preds = np.mean(multi_preds, 0)

preds = np.exp(preds)
np.save('test_TTA.npy', preds)
preds = np.load('test_TTA.npy')
print(np.shape(preds))
preds
print(np.max(preds))
print(np.min(preds))
np.shape(multi_preds)

def getBiggest(arr, n, axis=0):
    biggest = []
    for i in range(0, n):
        mx = np.argmax(arr, axis=axis)
        biggest.append(mx)
        arr = np.delete(arr, mx)
    return np.array(biggest)
    
def get_k_best_pred(preds):
    bigs = np.array([])
    for p in preds:
        bigs = np.append(bigs, getBiggest(p, 5))
    return np.reshape(bigs,(np.shape(preds)[0], 5))
k_best_pred = get_k_best_pred(preds)
pred_strings = k_best_pred.astype(str)
def pred_to_concat_str(pred_strings):
    pred_strings = k_best_pred.astype(str)
    for i, preds in enumerate(pred_strings):
        for j, classes in enumerate(preds):
            pred_strings[i,j] = data.classes[int(float(classes))]
    return np.array([' '.join(i) for i in pred_strings])
conc_pred_strings = pred_to_concat_str(pred_strings)
print(pred_strings)

ids = np.array(os.listdir(PATH+"/test"))
result = {'Id':conc_pred_strings}
submission = pd.DataFrame(result, index=ids)
submission.index.name = 'Image'
print(submission)
submission.to_csv('../working/submission.csv')