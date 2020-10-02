# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from fastai import *
from fastai.vision import *
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import shutil
np.random.seed(786)
ROOT = "/tmp/data"

def read_data(root):
    data = pd.read_csv(str(Path(root) / "VesselClassification.dat"), header=None, names=["image", "flag", "labelid", "label"], index_col=None)
    #data["labelid"] = data["label"].astype("category").cat.codes
    data["image"] = data.apply(lambda x: str(Path(str(x["labelid"])) / "{}.jpg".format(x["image"])), axis=1)
    data["image_exists"] = data["image"].apply(lambda x: (Path("/tmp/data/images/images") / x).exists())
    #print(data.head())
    data = data.loc[data["image_exists"]]
    data["flag"] = data["flag"] == 2
    return data
    
    
#Path(ROOT).mkdir(exist_ok=True, parents=True)
src1 = "../input/marvel-label-file/VesselClassification.dat"
src2 = "../input/marvel-maritime-vessels-classification-dataset"
shutil.copytree(src2, ROOT)
shutil.copy(src1, "/tmp/data/VesselClassification.dat")

if __name__=="__main__":
    data = read_data(ROOT)
    # data = data.sample(1000)
    print(data.shape)
    
    tfms1 = ([RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0.2, 0.8), 'col_pct': (0, 1),
                                                            'padding_mode': 'reflection'}, p=1.0, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmAffine (flip_affine), kwargs={}, p=0.5, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}, p=0.75, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmPixel (cutout), kwargs={'n_holes': (1, 10), 'length': (4, 10)}, p=0.75, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmAffine (rotate), kwargs={'degrees': (-20.0, 20.0)}, p=0.75, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.35), 'row_pct': (0.2, 0.8), 'col_pct': (0, 1)}, p=0.75, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmLighting (brightness), kwargs={'change': (0.4, 0.6)}, p=0.75, resolved={}, do_run=True, is_random=True),
              RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.8, 1.25)}, p=0.75, resolved={}, do_run=True, is_random=True)],
             [RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True)])
             
    src = (ImageList.from_df(data, path=str(Path(ROOT) / "images"), folder="images", cols=0).split_from_df(col=1).label_from_folder())

    data = ImageDataBunch.create_from_ll(src, ds_tfms=tfms1, size=(220, 330), bs=48)
    learn = cnn_learner(data, models.densenet169, metrics=accuracy, ps=0.6)
    learn.unfreeze()
    cb = callbacks.SaveModelCallback(learn, name="bestmodel")
    learn.fit_one_cycle(1, max_lr=3e-3, callbacks=[cb])
    learn.fit_one_cycle(2, max_lr=3e-3, callbacks=[cb])
    learn.fit_one_cycle(2, max_lr=3e-4, callbacks=[cb])

    fname = "bestmodel.pth"
    src = str(Path(ROOT) / "images" / "models" / fname)
    shutil.copy(src, fname)


    
    

    
    