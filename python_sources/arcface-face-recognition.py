#!/usr/bin/env python
# coding: utf-8

# In this notebook, you will see a pre-triend arcface model predicting results for every face in [pins face recognition dataset](https://www.kaggle.com/hereisburak/pins-face-recognition/) dataset. Then, we will compare those results through TensorBoard Projector and understands the idea behind face recognition.
# 
# If you want to train your own ArcFace model, please visit the open source [Liyana Face Analysis Project](https://github.com/aangfanboy/liyana/)
# 
# If you are interested at face processing(recognition, age-sex-ethnicty detection vs.), i higly recommend you to check [Liyana](https://github.com/aangfanboy/liyana/)
# 
# ![Liyana Image](https://raw.githubusercontent.com/aangfanboy/liyana/master/images-and-figures/liyana_1-2-0.png)

# I would like to drop a screenshot of what we are trying to achive in this notebook. 
# 
# In first screenshot, features of Anne Hathaway which extracted by *ArcFace model* and you can see that the **closest features to choosen Anne Hathaway face are also the features of Anne Hathaway.** That is the main idea of deep learning face recognition, **model extracts features and it designed to extract close features to same persons faces.**
# 
# ![Anne Hathaway](https://raw.githubusercontent.com/aangfanboy/liyana/master/images-and-figures/pins_tb_example.jpg)

# And last, i would like to show you more examples of face recognition that i like to share. Those videos also created by [Liyana](https://github.com/aangfanboy/liyana/)
# 
# 
# ### My favourite professor Gilbert Strang and My favourite podcaster Lex Fridman 
# ![gif1](https://github.com/aangfanboy/liyana/blob/master/images-and-figures/gil_lex_gif.gif?raw=true)
# 
# 
# ### Elon Musk and Joe Rogan
# ![gif2](https://github.com/aangfanboy/liyana/blob/master/images-and-figures/joe_elon_gif.gif?raw=true)
# 
# Since my show-off is done :) we can start.

# In[ ]:


import csv
import tensorflow as tf

from tqdm import tqdm


# In[ ]:


tf.__version__  # make sure you are using TensorFlow 2.x


# In[ ]:


model = tf.keras.models.load_model("../input/arcface-final-lresnet50ir/arcface_final.h5")


# In[ ]:


print(f"Input shape --> {model.input_shape}\nOutput Shape --> {model.output_shape}")

"""
As you can see, model take image(s) in 112x112 and with 3 channels(RGB, not BGR)
"""


# In[ ]:


# now i will define a data loader to read pins face recognition dataset, i will be receiving this code from Liyana

class DataEngineTypical:
    def make_label_map(self):
        self.label_map = {}

        for i, class_name in enumerate(tf.io.gfile.listdir(self.main_path)):
            self.label_map[class_name] = i

        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def path_yielder(self):
        for class_name in tf.io.gfile.listdir(self.main_path):
            if not "tfrecords" in class_name:
                for path_only in tf.io.gfile.listdir(self.main_path + class_name):
                    yield (self.main_path + class_name + "/" + path_only, self.label_map[class_name])

    def image_loader(self, image):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (112, 112), method="nearest")
        image = tf.image.random_flip_left_right(image)

        return (tf.cast(image, tf.float32) - 127.5) / 128.

    def mapper(self, path, label):
        return (self.image_loader(path), label)

    def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1,
                 reshuffle_each_iteration: bool = False, test_batch=64,
                 map_to: bool = True):
        self.main_path = main_path.rstrip("/") + "/"
        self.make_label_map()

        self.dataset_test = None
        if test_batch > 0:
            reshuffle_each_iteration = False
            print(f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")

        self.dataset = tf.data.Dataset.from_generator(self.path_yielder, (tf.string, tf.int64))
        if buffer_size > 0:
            self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

        if map_to:
            self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

        if test_batch > 0:
            self.dataset_test = self.dataset.take(int(test_batch))
            self.dataset = self.dataset.skip(int(test_batch))

        self.dataset = self.dataset.repeat(epochs)


# In[ ]:


# now i will define an Engine Class to run jobs for getting outputs from images

# this class will take data and get 512-D features through ArcFace model
class Engine:
    @staticmethod
    def flip_batch(batch):
        return batch[:, :, ::-1, :]

    def __init__(self, data_engine: DataEngineTypical):
        self.data_engine = data_engine
        self.model = tf.keras.models.load_model("../input/arcface-final-lresnet50ir/arcface_final.h5")

        tf.io.gfile.mkdir("projector_tensorboard")

    def __call__(self, flip: bool = False):
        metadata_file = open('projector_tensorboard/metadata.tsv', 'w')
        metadata_file.write('Class\tName\n')
        with open("projector_tensorboard/feature_vecs.tsv", 'w') as fw:
            csv_writer = csv.writer(fw, delimiter='\t')

            for x, y in tqdm(self.data_engine.dataset):
                outputs = self.model(x, training=False)
                if flip:
                    outputs += self.model(self.flip_batch(x), training=False)

                csv_writer.writerows(outputs.numpy())
                for label in y.numpy():
                    name = self.data_engine.reverse_label_map[label]
                    metadata_file.write(f'{label}\t{name}\n')

        metadata_file.close()


# In[ ]:


# Now we have both data loader and an engine to process that data through arcface model and save those features to a tsv file.

TDOM = DataEngineTypical(
    "../input/pins-face-recognition/105_classes_pins_dataset/",
    batch_size=64,
    epochs=1,
    buffer_size=0,
    reshuffle_each_iteration=False,
    test_batch=0
)  # TDOM for "Tensorflow Dataset Object Manager"

e = Engine(
    data_engine=TDOM
)

e()


# * In outputs, there should be folder named *projector_tensorboard*, that folder has two files named **feature_vecs.tsv** and **metadata.tsv**, download them.
# * Go [this](https://projector.tensorflow.org/) website, press **Load** in left, upload **feature_vecs.tsv** in **step 1** and upload **metadata.tsv** in **step 2**
# * Check *color by* to *class* and check the checkbox that says *Use categorical coloring*
# 
# If you did everything right, you should see a page like this:
# 
# ![page](https://raw.githubusercontent.com/aangfanboy/liyana/master/images-and-figures/pins_tb_mf_example.jpg)
# 
# 
# There are 17472 points that belongs to total 105 different human! And if you check carefully, you will discover the distance between males and females :) Because it is the most basic diversity of human face. Click one of the points and you will see the nearest point to that in the right side of page. Have fun :)

# In[ ]:




