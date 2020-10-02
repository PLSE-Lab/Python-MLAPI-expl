# %% [code]
!pip install bert-tensorflow
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
import bert
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Initialize session
sess = tf.Session()


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):

    train_df = pd.read_csv('../input/comments/train.csv')
    test_df = pd.read_csv('../input/testdata/test.csv')
    
    print(train_df)
    return train_df, test_df


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = [0]*21
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels),
    )


def convert_text_to_examples(texts, labels=None):
    """Create InputExamples"""
    InputExamples = []
    if not labels:
        for text in texts:
            InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=0)
        )
    else:
        for text, label in zip(texts, labels):
            InputExamples.append(
                InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
            )
    return InputExamples


class BERT(tf.keras.layers.Layer):
    def __init__(self, finetune_cells, debug=False, **kwargs):
        self.finetune_cells = finetune_cells
        self.trainable = True
        self.output_size = 768
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.debug = debug
        super(BERT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path,
                               trainable=self.trainable,
                               name="{}_module".format(self.name))
        
        trainable_vars = self.bert.variables

        t_vs = [var for var in trainable_vars if not "/cls/" in var.name]

        trainable_vars = t_vs

        layer_name_list = []

        for i, var in enumerate(trainable_vars):
            if self.debug:
                var_shape = var.get_shape()
                var_params = 1
                for dim in var_shape:
                    var_params *= dim
                print(str(i), "-", "var:", var.name)
                print(" ", "shape:", var_shape , "param:", var_params)
                
            if "layer" in var.name:
                layer_name = var.name.split("/")[3]
                layer_name_list.append(layer_name)

        layer_names = list(set(layer_name_list))
        layer_names.sort()

        if self.debug:
            print(layer_names)

        if self.finetune_cells == -1:
            for var in trainable_vars:
                if "/pooler/" in var.name:
                    # ignore the undocumented pooling layer
                    # we will create our own
                    pass
                else:
                    self._trainable_weights.append(var)

        else:
            # Select how many layers to fine tune
            last_n_layers = len(layer_names) - self.finetune_cells

            for var in trainable_vars:
                if "layer" in var.name:
                    layer_name = var.name.split("/")[3]
                    layer_num = int(layer_name.split("_")[1])+1
                    if layer_num > last_n_layers:
                        # Add to trainable weights
                        self._trainable_weights.append(var)

            if self.debug:
                print("BERT module loaded with", len(layer_names),
                    "Transformer cells, training all cells >", last_n_layers)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BERT, self).build(input_shape)

    def call(self, inputs):
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=tf.cast(input_ids, dtype="int32"),
                           input_mask=tf.cast(input_mask, dtype="int32"),
                           segment_ids=tf.cast(segment_ids, dtype="int32"))
        result = self.bert(inputs=bert_inputs,
                           signature="tokens",
                           as_dict=True)["sequence_output"]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BERT(finetune_cells=1,
              debug=False)(bert_inputs)
    dense = tf.keras.layers.GlobalMaxPooling1D()(bert_output)
    pred = tf.keras.layers.Dense(21, activation="softmax")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def main():
    # Params for bert model and tokenization
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    max_seq_length = 256

    train_df, test_df = download_and_load_datasets()

    le = LabelEncoder()
    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df["Review Text"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_df['polarity'] = le.fit_transform(train_df['topic'])
    train_label = to_categorical(train_df['polarity'].values).tolist()
    #train_label = train_df["polarity"].tolist()
    
    test_text = test_df["Review Text"].tolist()
    test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    #test_df['polarity'] = le.transform(test_df['topic'])
    #test_label = to_categorical(test_df['polarity'].values).tolist()
    #test_label = test_df["polarity"].values

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path)

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text)

    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_seq_length
    )
    (
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels,
    ) = convert_examples_to_features(
        tokenizer, test_examples, max_seq_length=max_seq_length
    )

    model = build_model(max_seq_length)

    # Instantiate variables
    initialize_vars(sess)
    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_labels,
        epochs=3,
        batch_size=32,
    )
    prediction = model.predict([test_input_ids, test_input_masks, test_segment_ids])
    print(prediction)
    indexes = tf.argmax(prediction, axis=1)
    #print(indexes.eval(session=sess))
    #acc = accuracy_score(test_df['polarity'].values,indexes.eval(session=sess))
    #print(acc)
    test_df['topic'] = le.inverse_transform(indexes.eval(session=sess))
    print(test_df)
    test_df.to_csv("output.csv")


if __name__ == "__main__":
    main()