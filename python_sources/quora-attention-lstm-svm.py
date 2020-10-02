import tensorflow as tf
import numpy as np 
import math

from collections import Counter
import re

import csv
import os

DATA_PATH = "../input/"
MODEL_PATH = "./"
USE_GPU = tf.test.is_gpu_available()
BATCH_SIZE = 128


def build_cell(num_layers, num_hidden, keep_prob, use_cuda=False):
    cells = []

    for _ in range(num_layers):
        if use_cuda:
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(
                num_hidden,
                forget_bias=0.0,
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
            )

        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cells.append(cell)

    if num_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(cells)
    return cells[0]
    
    
def sum_reduce(seq_fw, seq_bw):
    if tf.contrib.framework.nest.is_sequence(seq_fw):
        tf.contrib.framework.nest.assert_same_structure(seq_fw, seq_bw)

        x_flat = tf.contrib.framework.nest.flatten(seq_fw)
        y_flat = tf.contrib.framework.nest.flatten(seq_bw)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(tf.add_n([x_i, y_i]))

        return tf.contrib.framework.nest.pack_sequence_as(seq_fw, flat)
    return tf.add_n([seq_fw, seq_bw])


def concat_reducer(seq_fw, seq_bw):
    if tf.contrib.framework.nest.is_sequence(seq_fw):
        tf.contrib.framework.nest.assert_same_structure(seq_fw, seq_bw)

        x_flat = tf.contrib.framework.nest.flatten(seq_fw)
        y_flat = tf.contrib.framework.nest.flatten(seq_bw)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(tf.concat([x_i, y_i], axis=-1))

        return tf.contrib.framework.nest.pack_sequence_as(seq_fw, flat)
    return tf.concat([seq_fw, seq_bw], axis=-1)
    

class AttentionSVM(object):

    def __init__(self, input_x, vocab_size, embedding_size, keep_prob, num_hidden, attention_size,
                 is_training=False, input_y=None):
        self._batch_size, self._time_steps = input_x.get_shape().as_list()
        self._num_hidden = num_hidden
        self._attention_size = attention_size

        self._use_cuda = tf.test.is_gpu_available()

        self._input_keep_prob, self._lstm_keep_prob, self._dense_keep_prob = tf.unstack(keep_prob)

        # input embedding
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self._seq_lengths = tf.count_nonzero(input_x, axis=1, name="sequence_lengths")
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="word_embeddings",
                dtype=tf.float32,
                trainable=True
            )
            self._input = tf.nn.embedding_lookup(embeddings, input_x)
            input_x = self.__input_op()

        # bi-directional encoder
        self.__context, final_state, shape = self.__encoder(input_x)

        # dense layer
        self.__output = self.__dense(self.__context)
        self.__predictions = tf.sign(tf.maximum(self.__output, 0))

        if is_training:
            assert input_y is not None
            self.target = input_y * 2 - 1
            self._input = tf.stack(self._input)

            with tf.variable_scope('cost'):
                self.loss = self.__cost()

            with tf.variable_scope('optimizer'):
                self._lr = tf.Variable(0.0, trainable=False)
                self._clip_norm = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
                opt = tf.train.GradientDescentOptimizer(self._lr)

                # compute the gradient norm - only for logging purposes - remove if greatly affecting performance
                self.gradient_norm = tf.sqrt(sum([tf.norm(t) ** 2 for t in grads]), name="gradient_norm")

                self.train = opt.apply_gradients(
                    zip(grads, t_vars),
                    global_step=tf.train.get_or_create_global_step()
                )

                self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                self._lr_update = tf.assign(self._lr, self._new_lr)

                self._new_clip_norm = tf.placeholder(tf.float32, shape=[], name="new_clip_norm")
                self._clip_norm_update = tf.assign(self._clip_norm, self._new_clip_norm)

    def assign_lr(self, session, lr_value):
        """
        Updates the learning rate
        :param session: (TensorFlow Session)
        :param lr_value: (float)
        """

        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_clip_norm(self, session, norm_value):
        """
        Updates the gradient normalization factor
        :param session: (TensorFlow Session)
        :param norm_value: (float)
        """

        session.run(self._clip_norm_update, feed_dict={self._new_clip_norm: norm_value})

    def __input_op(self):
        with tf.variable_scope('input_dropout', reuse=tf.AUTO_REUSE):
            input_x = self._input
            return tf.nn.dropout(x=input_x, keep_prob=self._input_keep_prob)

    def __encoder(self, input_x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            num_layers = 2

            forward = build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self._use_cuda
            )
            backward = build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self._use_cuda
            )

            output_seq, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward,
                cell_bw=backward,
                inputs=input_x,
                dtype=tf.float32,
                sequence_length=self._seq_lengths
            )

            # perform element-wise summations to combine the forward and backward sequences
            encoder_output = concat_reducer(seq_fw=output_seq[0], seq_bw=output_seq[1])
            encoder_state = final_state

            # get the context vectors from the attention mechanism
            context = self.__attention(input_x=encoder_output)

            return context, encoder_state, tf.shape(encoder_output)

    def __attention(self, input_x):
        with tf.variable_scope('source_attention', reuse=tf.AUTO_REUSE):
            in_dim = input_x.get_shape().as_list()[-1]
            w_omega = tf.get_variable(
                "w_omega",
                shape=[in_dim, self._attention_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            b_omega = tf.get_variable("b_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())

            v = tf.tanh(tf.einsum("ijk,kl->ijl", input_x, w_omega) + b_omega)
            vu = tf.einsum("ijl,l->ij", v, u_omega, name="Bahdanau_score")
            alphas = tf.nn.softmax(vu, name="attention_weights")

            output = tf.reduce_sum(input_x * tf.expand_dims(alphas, -1), 1, name="context_vector")
            return output

    def __dense(self, input_x):
        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            input_dim = input_x.get_shape().as_list()[-1]
            input_x = tf.nn.dropout(input_x, keep_prob=self._dense_keep_prob, name="dense_dropout")

            weight = tf.get_variable(
                "dense_weight",
                shape=[input_dim],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(-0.01, 0.01)
            )

            bias = tf.get_variable(
                "dec_bias",
                shape=[],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            output = tf.tensordot(input_x, weight, axes=1) - bias
            return output

    def __cost(self):
        with tf.variable_scope('linear_svm_loss', reuse=tf.AUTO_REUSE):
            projection_dist = 1 - self.target * self.__output
            margin = tf.nn.relu(projection_dist)
            loss = tf.reduce_mean(margin ** 2)

        with tf.variable_scope('l2_loss', reuse=tf.AUTO_REUSE):
            weights = tf.trainable_variables()

            # only perform L2-regularization on the fully connected layer(s)
            l2_losses = [tf.nn.l2_loss(v) for v in weights if 'dense_weight' in v.name]
            loss += 1e-3 * tf.add_n(l2_losses)
        return loss

    @property
    def lr(self):
        return self._lr

    @property
    def clip_norm(self):
        return self._clip_norm

    @property
    def margin_distance(self):
        return self.__output

    @property
    def predict(self):
        return self.__predictions
        
        
class EmbeddingLookup(object):

    def __init__(self, top_n=None, use_tf_idf_importance=False):
        self._tokenizer = re.compile("\w+|\$[\d\.]+").findall
        self._top_n = top_n
        self._use_tf_idf = use_tf_idf_importance

        self._dictionary = None
        self._reverse = None
        self._unknown = "<unk>"

    def _get_top_n_tokens_tf_idf(self, corpus):
        # create set of tokens for each document
        doc_lookups = [set(self._tokenizer(document)) for document in corpus]

        inverse_df = []
        for i in range(len(corpus)):
            document = doc_lookups[i]

            # for each document, find all of the unique words
            # these will be counted up at the end (unique occurrences) to get document frequencies
            present_tokens = document.intersection(self._dictionary)
            inverse_df.extend(present_tokens)

        # compute the smooth TF-IDF values
        inverse_df = {k: math.log(1 + (1 + len(corpus)) / (1 + v)) for k, v in Counter(inverse_df).items()}
        total_dfs = sum(self._dictionary.values())
        tf_idf = {token: (self._dictionary[token] / total_dfs) * inverse_df[token] for token in inverse_df}

        # sort by TF-IDF values, descending, then build the dictionary using these tokens
        sort_terms = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        top = {item[0]: idx + 1 for idx, item in enumerate(sort_terms)}
        top[self._unknown] = len(top) + 1
        return top

    def _get_top_n_tokens(self):
        top = self._dictionary.most_common(self._top_n)
        top = {item[0]: idx + 1 for idx, item in enumerate(top)}
        top[self._unknown] = len(top) + 1
        return top

    def fit(self, corpus):
        tokens = [word for text in corpus for word in self._tokenizer(text.strip())]
        self._dictionary = Counter(tokens)

        if self._top_n is not None and self._top_n < len(self._dictionary):
            if self._use_tf_idf:
                self._dictionary = self._get_top_n_tokens_tf_idf(corpus)
            else:
                self._dictionary = self._get_top_n_tokens()
        else:
            self._dictionary = {item[0]: idx + 1 for idx, item in enumerate(self._dictionary.items())}

            # add the <unk> token to the embedding lookup
            size = len(self._dictionary)
            self._dictionary[self._unknown] = size + 1

        self._reverse = {v: k for k, v in self._dictionary.items()}
        return self

    def transform(self, corpus):
        if self._dictionary is None:
            raise AttributeError("You must fit the lookup first, either by calling the fit or fit_transform methods.")

        corpus = [
            [
                self._dictionary[word] if word in self._dictionary else self._dictionary[self._unknown]
                for word in self._tokenizer(text.strip())
            ] for text in corpus]
        return corpus

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def __getitem__(self, key):
        if not isinstance(self._dictionary, dict):
            raise TypeError("You must fit the lookup first, either by calling the fit or fit_transform methods.")
        return self._dictionary.get(key, None)

    @property
    def reverse(self):
        return self._reverse
        
        
def log(message):
    out = f"[INFO] {message}"
    print(out)


def pre_process(text):
    text = text.strip()

    # remove URLs
    text = re.sub(r"^https?://.*[\r\n]*", "", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"http\S+(\s)*(\w+\.\w+)*", "", text, re.MULTILINE | re.IGNORECASE)

    # un-contract
    text = re.sub(r"\'ve", " have ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"cant't", " cannot ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"n't", " not ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"I'm", " I am ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'re", " are ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'d", " would ", text, re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\'ll", " will ", text, re.MULTILINE | re.IGNORECASE)

    # pad punctuation marks
    text = re.sub(r"([!\"#\$%&\'\(\)\*\+,-\.\/:;\<\=\>\?@\[\\\]\^_`\{\|\}~])", r" \1", text, re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text, re.MULTILINE)

    return text


def load_text(infer=False):
    path = DATA_PATH
    files = ["train.csv"]
    if infer:
        files = ["test.csv"]
    ids, texts, targets = [], [], []

    for file in files:
        with open(path + file, "r", encoding="latin1") as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                ids.append(row["qid"])
                texts.append(pre_process(row["question_text"]))
                targets.append(int(row.get("target", -1)))

    return np.array(ids), np.array(texts), np.array(targets, np.float32)


def sub_sample(data, split_type='balanced', skew=1):
    ids, texts, targets = data
    assert isinstance(targets, np.ndarray)

    size_pos_class = targets[targets == 1].shape[0]
    size_neg_class = targets[targets == 0].shape[0]
    assert split_type in {'balanced', 'skew'}
    if split_type == 'skew':
        assert isinstance(skew, int) and skew > 0

    if split_type == 'balanced':
        indices = np.random.permutation(range(size_neg_class))[:size_pos_class]
    elif split_type == 'skew':
        assert isinstance(skew, int) and skew > 0
        indices = np.random.permutation(range(size_neg_class))[:skew * size_pos_class]
    else:
        raise ValueError

    ids_ = np.concatenate((ids[targets == 1], ids[targets == 0][indices]))
    texts_ = np.concatenate((texts[targets == 1], texts[targets == 0][indices]))
    targets_ = np.concatenate((targets[targets == 1], targets[targets == 0][indices]))
    return ids_, texts_, targets_


def test_val_split(corpus, val_size):
    ids, texts, targets = corpus
    s = np.random.permutation(range(len(ids)))

    cv_ids = ids[s[:val_size]]
    cv_texts = texts[s[:val_size]]
    cv_targets = targets[s[:val_size]]

    train_ids = ids[s[val_size:]]
    train_texts = texts[s[val_size:]]
    train_targets = targets[s[val_size:]]
    return (train_ids, train_texts, train_targets), (cv_ids, cv_texts, cv_targets)


def remove_null_sequences(ids, sequences, targets):
    ids_, sequences_, targets_ = [], [], []
    for i in range(len(ids)):
        seq = sequences[i]
        if seq and len(seq) > 0:
            ids_.append(ids[i])
            sequences_.append(sequences[i])
            targets_.append(targets[i])
    return np.array(ids_), sequences_, np.array(targets_, dtype=np.float32)
    
    
def pad_sequence(sequence, max_sequence_length):
    """
    Pads individual text sequences to the maximum length
    seen by the model at training time
    :param sequence: list of integer lookup keys for the vocabulary (list)
    :param max_sequence_length: (int)
    :return: padded sequence (ndarray)
    """

    sequence = np.array(sequence, dtype=np.int32)
    difference = max_sequence_length - sequence.shape[0]
    pad = np.zeros((difference,), dtype=np.int32)
    return np.concatenate((sequence, pad))


def f1(predicted, expected):
    tp = np.sum(predicted * expected)
    fp = np.sum(predicted * (1 - expected))
    fn = np.sum((1 - predicted) * expected)

    precision = tp / (tp + fp) if tp + fp > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_score
    
    
def mini_batches(corpus, size, n_batches, max_len, seed):
    np.random.seed(seed)
    sequences, targets = corpus
    s = np.random.choice(range(len(targets)), replace=False, size=min(len(targets), size * n_batches)).astype(np.int32)

    for mb in range(n_batches):
        mini_batch = s[mb * size: (mb + 1) * size]
        x = np.array([pad_sequence(sequences[index], max_len) for index in mini_batch])
        y = targets[mini_batch]
        yield x, y
        
        
def train(model_folder, num_tokens=10000, num_hidden=128, attention_size=128,
          batch_size=32, num_batches=50, num_epochs=10,
          use_tf_idf=False):

    log_dir = MODEL_PATH + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log("Fetching corpus and transforming to frequency domain")
    corpus = load_text()

    log("Sub-sampling to balance classes")
    corpus = sub_sample(data=corpus, split_type="skew", skew=3)

    log("Splitting the training and validation sets")
    train_data, cv_data = test_val_split(corpus=corpus, val_size=512)

    t_ids, t_texts, t_targets = train_data
    cv_ids, cv_texts, cv_targets = cv_data

    log("Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens, use_tf_idf_importance=use_tf_idf)
    full_text = lookup.fit_transform(corpus=t_texts)
    cv_x = lookup.transform(corpus=cv_texts)

    log("Removing empty sequences")
    t_ids, full_text, t_targets = remove_null_sequences(ids=t_ids, sequences=full_text, targets=t_targets)
    cv_ids, cv_x, cv_targets = remove_null_sequences(ids=cv_ids, sequences=cv_x, targets=cv_targets)

    log("Getting the maximum sequence length and vocab size")
    max_seq_len = max([len(seq) for seq in full_text + cv_x])
    vocab_size = max([max(seq) for seq in full_text + cv_x]) + 1

    log(f"Padding sequences in corpus to length {max_seq_len}")
    full_text = np.array([pad_sequence(seq, max_seq_len) for seq in full_text])
    cv_x = np.array([pad_sequence(seq, max_seq_len) for seq in cv_x])
    keep_probabilities = [1.0, 0.7, 1.0]

    log("Compiling seq2seq automorphism model")
    seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len])
    target_input = tf.placeholder(dtype=tf.float32, shape=[None, ])
    keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

    model = AttentionSVM(
        input_x=seq_input,
        embedding_size=512,
        vocab_size=vocab_size,
        keep_prob=keep_prob,
        num_hidden=num_hidden,
        attention_size=attention_size,
        is_training=True,
        input_y=target_input
    )

    lstm_file_name = None
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8,
        allow_growth=True
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        model.assign_lr(sess, 0.1)
        model.assign_clip_norm(sess, 10.0)

        for epoch in range(num_epochs):
            print("\t Epoch: {0}".format(epoch + 1))
            i = 1

            for x, y in mini_batches(
                    (full_text, t_targets),
                    size=batch_size,
                    n_batches=num_batches,
                    max_len=max_seq_len,
                    seed=epoch
            ):
                if x.shape[0] == 0:
                    continue

                loss_val, gradient, _ = sess.run(
                    [model.loss, model.gradient_norm, model.train],
                    feed_dict={
                        seq_input: x,
                        target_input: y,
                        keep_prob: keep_probabilities
                    }
                )

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(i, loss_val))
                i += 1

            cv_loss, decision, predictions = sess.run(
                [model.loss, model.margin_distance, model.predict],
                feed_dict={seq_input: cv_x, target_input: cv_targets, keep_prob: keep_probabilities}
            )
            f1_score = f1(predicted=predictions, expected=cv_targets)
            log(f"Cross-validation F1-score: {f1_score}")

        log("Running inferencing and outputing to submission.csv")
        ids, texts, _ = load_text(infer=True)
        sequences = lookup.transform(corpus=texts)
        sequences = np.array([pad_sequence(seq, max_seq_len) for seq in sequences])
        
        fieldnames = ['qid', 'prediction']
        csv_f = open(MODEL_PATH + "submission.csv", "w")
        writer = csv.DictWriter(csv_f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(sequences) + BATCH_SIZE, BATCH_SIZE):
            seq = sequences[i: i + BATCH_SIZE]
            qids = ids[i: i + BATCH_SIZE]
            
            if seq.shape[0] == 0:
                continue
            predictions = sess.run(model.predict, feed_dict={seq_input: seq})
            rows = [{
                "qid": qid,
                "prediction": int(prediction)
            } for qid, prediction in zip(qids, predictions)]
            writer.writerows(rows)

        csv_f.close()

    return lstm_file_name
    

if __name__ == '__main__':
    name = "quora_svm_v1"

    train(
        model_folder=name,
        num_tokens=100000,
        num_hidden=64,
        attention_size=32,
        batch_size=64,
        num_batches=1000,
        num_epochs=25,
        use_tf_idf=False
    )
