#!/usr/bin/env python


import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers.core import Dense
from tqdm import tqdm, trange
#################################################################################################################

###############################################################################################################################
#!/usr/bin/env python

import os

import numpy as np

_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)


def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    return _idx_to_word[token]


embeddings_path = os.path.join( '../input/glove.6B.100d.txt')
with open(embeddings_path) as f:
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    vocab_size = sum(1 for line in f)
    vocab_size += 3
    f.seek(0)

    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    glove[UNKNOWN_TOKEN] = np.zeros(dimensions)
    glove[START_TOKEN] = -np.ones(dimensions)
    glove[END_TOKEN] = np.ones(dimensions)

    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break
##################################################################################################################################
#################################################################################################################################
#!/usr/bin/env python

from collections import Counter

import csv



_MAX_BATCH_SIZE = 128


def _tokenize(string):
    return [word.lower() for word in string.split(" ")]


def _prepare_batch(batch):
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    question_text = []
    question_input_words = []
    question_output_words = []
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])

        question_words = entry["question_words"]
        question_input_words.append([START_WORD] + question_words)
        question_output_words.append(question_words + [END_WORD])

    batch_size = len(batch)
    max_document_len = max(len(document) for document in document_words)
    max_answer_len = max(len(answer) for answer in answer_indices)
    max_question_len = max(len(question) for question in question_input_words)

    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, index] = 1
            answer_masks[i, j, index] = 1
        answer_lengths[i] = len(answer_indices[i])

        for j, word in enumerate(question_input_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words[i])

    return {
        "size": batch_size,
        "document_ids": document_ids,
        "document_text": document_text,
        "document_words": document_words,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_text": answer_text,
        "answer_indices": answer_indices,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "question_text": question_text,
        "question_input_tokens": question_input_tokens,
        "question_output_tokens": question_output_tokens,
        "question_lengths": question_lengths,
    }


def collapse_documents(batch):
    seen_ids = set()
    keep = []

    for i in range(batch["size"]):
        id = batch["document_ids"][i]
        if id in seen_ids:
            continue

        keep.append(i)
        seen_ids.add(id)

    result = {}
    for key, value in batch.items():
        if key == "size":
            result[key] = len(keep)
        elif isinstance(value, np.ndarray):
            result[key] = value[keep]
        else:
            result[key] = [value[i] for i in keep]
    return result


def expand_answers(batch, answers):
    new_batch = []

    for i in range(batch["size"]):
        split_answers = []
        last = None
        for j, tag in enumerate(answers[i]):
            if tag:
                if last != j - 1:
                    split_answers.append([])
                split_answers[-1].append(j)
                last = j

        for answer_indices in split_answers:
            document_id = batch["document_ids"][i]
            document_text = batch["document_text"][i]
            document_words = batch["document_words"][i]
            answer_text = " ".join(document_words[i] for i in answer_indices)
            new_batch.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": "",
                "question_words": [],
            })

    return _prepare_batch(new_batch)


def _read_data(path):
    stories = {}

    with open(path) as f:
        header_seen = False
        for row in csv.reader(f):
            if not header_seen:
                header_seen = True
                continue

            document_id = row[0]

            existing_stories = stories.setdefault(document_id, [])

            document_text = row[1]
            if existing_stories and document_text == existing_stories[0]["document_text"]:
                # Save memory by sharing identical documents
                document_text = existing_stories[0]["document_text"]
                document_words = existing_stories[0]["document_words"]
            else:
                document_words = _tokenize(document_text)

            question_text = row[2]
            question_words = _tokenize(question_text)

            answer = row[3]
            answer_indices = []
            for chunk in answer.split(","):
                start, end = (int(index) for index in chunk.split(":"))
                answer_indices.extend(range(start, end))
            answer_text = " ".join(document_words[i] for i in answer_indices)

            existing_stories.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": question_text,
                "question_words": question_words,
            })

    return stories


def _process_stories(stories):
    batch = []
    for story in stories.values():
        if len(batch) + len(story) > _MAX_BATCH_SIZE:
            yield _prepare_batch(batch)
            batch = []
        batch.extend(story)
    if batch:
        yield _prepare_batch(batch)


_training_stories = None
_test_stories = None

def _load_training_stories():
    global _training_stories
    if not _training_stories:
        _training_stories = _read_data("../input/train.csv")
    return _training_stories

def _load_test_stories():
    global _test_stories
    if not _test_stories:
        _test_stories = _read_data("../input/test.csv")
    return _test_stories

def training_data():
    return _process_stories(_load_training_stories())

def test_data():
    return _process_stories(_load_test_stories())


def trim_embeddings():
    document_counts = Counter()
    question_counts = Counter()
    for data in [_load_training_stories().values(), _load_test_stories().values()]:
        for stories in data:
            document_counts.update(stories[0]["document_words"])
            for story in stories:
                question_counts.update(story["question_words"])

    keep = set()
    for word, count in question_counts.most_common(5000):
        keep.add(word)
    for word, count in document_counts.most_common():
        if len(keep) >= 10000:
            break
        keep.add(word)

    with open("glove.6B.100d.txt") as f:
        with open("glove.6B.100d.trimmed.txt", "w") as f2:
            for line in f:
                if line.split(" ")[0] in keep:
                    f2.write(line)

#############################################################################################################################
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_dir')

#prepare_embeddings_for_tensorboard(embedding_vis, log_dir)

##########ADD##########

hparams = tf.contrib.training.HParams(
    batch_size=5,
    num_units=10,
    beam_width =9,
    use_attention = False,
)
#############################
import itertools
embedding = tf.get_variable("embedding", initializer=glove)

EMBEDDING_DIMENS = glove.shape[1]


document_tokens = tf.placeholder(tf.int32, shape=[None, None], name="document_tokens")
document_lengths = tf.placeholder(tf.int32, shape=[None], name="document_lengths")

document_emb = tf.nn.embedding_lookup(embedding, document_tokens)

forward_cell = GRUCell(EMBEDDING_DIMENS)
backward_cell = GRUCell(EMBEDDING_DIMENS)

answer_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, document_emb, document_lengths, dtype=tf.float32, scope="answer_rnn")
answer_outputs = tf.concat(answer_outputs, 2)

answer_tags = tf.layers.dense(inputs=answer_outputs, units=2)

answer_labels = tf.placeholder(tf.int32, shape=[None, None], name="answer_labels")

answer_mask = tf.sequence_mask(document_lengths, dtype=tf.float32)
answer_loss = seq2seq.sequence_loss(logits=answer_tags, targets=answer_labels, weights=answer_mask, name="answer_loss")


encoder_input_mask = tf.placeholder(tf.float32, shape=[None, None, None], name="encoder_input_mask")
encoder_inputs = tf.matmul(encoder_input_mask, answer_outputs, name="encoder_inputs")
encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")

encoder_cell = GRUCell(forward_cell.state_size + backward_cell.state_size)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, encoder_lengths, dtype=tf.float32, scope="encoder_rnn")


decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
decoder_labels = tf.placeholder(tf.int32, shape=[None, None], name="decoder_labels")
decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")

decoder_emb = tf.nn.embedding_lookup(embedding, decoder_inputs)
helper = seq2seq.TrainingHelper(decoder_emb, decoder_lengths)

projection = Dense(embedding.shape[0], use_bias=False)

decoder_cell = GRUCell(encoder_cell.state_size)

#############ADD###############

if hparams.use_attention:
  # Attention
  # attention_states: [batch_size, max_time, num_units]
  attention_states = tf.transpose(encoder_outputs)

  # Create an attention mechanism
  attention_mechanism = tf.contrib.seq2seq.LuongAttention(
      hparams.num_units, attention_states,
      memory_sequence_length=None)

  decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
      decoder_cell, attention_mechanism,
      attention_layer_size=hparams.num_units)

  initial_state = decoder_cell.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)
else:
  initial_state = encoder_state

  ###############################


decoder = seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection)

decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, scope="decoder")
decoder_outputs = decoder_outputs.rnn_output

question_mask = tf.sequence_mask(decoder_lengths, dtype=tf.float32)
question_loss = seq2seq.sequence_loss(logits=decoder_outputs, targets=decoder_labels, weights=question_mask, name="question_loss")


loss = tf.add(answer_loss, question_loss, name="loss")
tf.summary.scalar("loss", loss)
optimizer = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()


saver = tf.train.Saver()
session = tf.InteractiveSession()
writer = tf.summary.FileWriter(log_dir, session.graph)

EPOCHS = 5

epoch = 0
for i in range(1, EPOCHS + 1):
    if os.path.exists("model-{0}.index".format(i)):
        epoch = i

if epoch:
    saver.restore(session, "model-{0}".format(epoch))
else:
    session.run(tf.global_variables_initializer())

batch_index = 0
batch_count = None
for epoch in trange(epoch + 1, EPOCHS + 1, desc="Epochs", unit="epoch"):
    batches = tqdm(training_data(), total=batch_count, desc="Batches", unit="batch")
    for batch in batches:
        _, loss_value, summary = session.run([optimizer, loss, merged], {
            document_tokens: batch["document_tokens"],
            document_lengths: batch["document_lengths"],
            answer_labels: batch["answer_labels"],
            encoder_input_mask: batch["answer_masks"],
            encoder_lengths: batch["answer_lengths"],
            decoder_inputs: batch["question_input_tokens"],
            decoder_labels: batch["question_output_tokens"],
            decoder_lengths: batch["question_lengths"],
        })
        batches.set_postfix(loss=loss_value)
        writer.add_summary(summary, batch_index)
        writer.flush()
        batch_index += 1

    if batch_count is None:
        batch_count = batch_index

    saver.save(session, "model", epoch)


batch = next(test_data())
batch = collapse_documents(batch)

answers = session.run(answer_tags, {
    document_tokens: batch["document_tokens"],
    document_lengths: batch["document_lengths"],
})
answers = np.argmax(answers, 2)


batch = expand_answers(batch, answers)

helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch["size"]], START_TOKEN), END_TOKEN)
decoder = seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection)
decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=16)
decoder_outputs = decoder_outputs.rnn_output

############ADD##########################

# Beam Search
# Replicate encoder infos beam_width times
decoder_initial_state = tf.contrib.seq2seq.tile_batch(
    initial_state, multiplier=hparams.beam_width)
# Define a beam-search decoder
inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=START_TOKEN,
        end_token=END_TOKEN,
        initial_state=decoder_initial_state,
        beam_width=hparams.beam_width,
        output_layer=projection,
        length_penalty_weight=0.0)

# Dynamic decoding
decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    inference_decoder, maximum_iterations=16)
decoder_outputs = decoder_outputs.predicted_ids

############################################

questions = session.run(decoder_outputs, {
    document_tokens: batch["document_tokens"],
    document_lengths: batch["document_lengths"],
    answer_labels: batch["answer_labels"],
    encoder_input_mask: batch["answer_masks"],
    encoder_lengths: batch["answer_lengths"],
})
questions[:,:,UNKNOWN_TOKEN] = 0
questions = np.argmax(questions, 2)

for i in range(batch["size"]):
    question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
    print("Question: " + " ".join(look_up_token(token) for token in question))
    print("Answer: " + batch["answer_text"][i])
    print()

###################################################################################################################################
# !/usr/bin/python
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class NewsQAData( object ):
    def __init__(self, filepath, max_words=None):

        print( 'Processing text dataset' )

        question_texts, answer_texts = [ ], [ ]

        for i, row in pd.read_csv( filepath ).iterrows():
            try:
                question = row[ 'question' ]
                stok, etok = [ int( i ) for i in row[ 'answer_token_ranges' ].split( ':' ) ]
                answer = ''.join( row[ 'story_text' ].split()[ stok:etok ] )
            except:
                # Skip bad token ranges
                continue
            question_texts.append( question )
            answer_texts.append( answer )

        # finally, vectorize the text samples into a 2D integer tensor
        # Answers
        self.answer_tokenizer = Tokenizer( num_words=max_words )
        self.answer_tokenizer.fit_on_texts( answer_texts )
        answer_sequences = self.answer_tokenizer.texts_to_sequences( answer_texts )
        self.answer_word_index = self.answer_tokenizer.word_index
        print( 'Found %s unique tokens in the answers' % len( self.answer_word_index ) )
        self.answer_data = pad_sequences( answer_sequences, maxlen=None )

        # Questions
        self.question_tokenizer = Tokenizer( num_words=max_words )
        self.question_tokenizer.fit_on_texts( question_texts )
        question_sequences = self.question_tokenizer.texts_to_sequences( question_texts )
        self.question_word_index = self.question_tokenizer.word_index
        print( 'Found %s unique tokens in the questions' % len( self.question_word_index ) )
        question_data_sparse = pad_sequences( question_sequences, maxlen=None )

        # Blow out to one hot encoding per word
        self.question_data = np.zeros( (question_data_sparse.shape[ 0 ],
                                        question_data_sparse.shape[ 1 ],
                                        len( self.question_word_index ) + 1),
                                       dtype=np.bool )
        for t in range( question_data_sparse.shape[ 0 ] ):
            for s in range( question_data_sparse.shape[ 1 ] ):
                v_i = question_data_sparse[ t, s ]
                self.question_data[ t, s, v_i ] = 1

        # Invert question word index for fast lookup
        self.question_index_to_word_map = {}
        for word, ix in self.question_word_index.items():
            self.question_index_to_word_map[ ix ] = word

        print( 'Shape of answer data tensor:', self.answer_data.shape )
        print( 'Shape of question data tensor:', self.question_data.shape )

    def get_answer_question_data(self):
        return self.answer_data, self.question_data

    def get_question_vocab_size(self):
        return len( self.question_word_index ) + 1

    def get_answer_vocab_size(self):
        return len( self.answer_word_index ) + 1

    def get_question_word_index(self):
        return self.question_word_index

    def get_answer_word_index(self):
        return self.answer_word_index

    def encode_answers(self, texts):
        return pad_sequences( self.answer_tokenizer.texts_to_sequences( texts ),
                              maxlen=self.answer_data.shape[ 1 ] )

    def decode_questions(self, encoded_questions, remove_oov=False, remove_padding=False):
        max_indices_per_time = np.argmax( encoded_questions, axis=2 )
        decoded_questions = [ ]
        for x in range( encoded_questions.shape[ 0 ] ):
            decoded_tokens = [ ]
            for t in range( encoded_questions.shape[ 1 ] ):
                word_ix = max_indices_per_time[ x, t ]
                if word_ix == 0 and not remove_oov:
                    decoded_tokens.append( "-UNK-" )
                else:
                    decoded_tokens.append( self.question_index_to_word_map[ word_ix ] )

            if remove_padding:
                start, end = 0, len( decoded_tokens )
                for token in decoded_tokens:
                    if token != "-UNK-":
                        break
                    start += 1
                for token in decoded_tokens[ ::-1 ]:
                    if token != "-UNK-":
                        break
                    end -= 1
                decoded_tokens = decoded_tokens[ start:end ]

            decoded_questions.append( " ".join( decoded_tokens ) + "?" )

        return decoded_questions
##############################################################################################################################
import io
import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm


# The code below is mostly from `https://www.tensorflow.org/programmers_guide/embedding`.

def prepare_embeddings_for_tensorboard(embeddings_path: str, log_dir: str,
                                       token_filter: Optional[set] = None):

    """
    Prepare embeddings for TensorBoard by writing them to `log_dir` in the required format.
    :param embeddings_path: The path to the GloVe embeddings file.
    :param log_dir: The directory for TensorBoard.
    :param token_filter: The set of tokens to use. If not given, then all tokens will be used.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.info("Loading embeddings from `%s`.", embeddings_path)

    metadata = []
    vectors = []

    with open(embeddings_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings",
                         mininterval=2, unit=" tokens", unit_scale=True):
            split = line.rstrip().split(' ')
            token = split[0]
            if token_filter is None or token in token_filter:
                metadata.append(token)
                vector = [float(x) for x in split[1:]]
                vectors.append(vector)

    assert len(vectors) > 0, "No vectors found."

    embeddings = np.array(vectors, dtype=np.float32)

    # Write metadata.
    metadata_path = os.path.join(log_dir, 'vocab.tsv')
    with io.open(metadata_path, 'w', encoding='utf-8') as f:
        for message in metadata:
            f.write(message)
            f.write("\n")

    embedding_var = tf.Variable(embeddings, name='message_embedding')

    with tf.Session() as sess:
        saver = tf.train.Saver([embedding_var])
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(log_dir, 'embeddings.ckpt'))

    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(log_dir, metadata_path)

    summary_writer = tf.summary.FileWriter(log_dir)

    projector.visualize_embeddings(summary_writer, config)

    logging.info("Run: `tensorboard --logdir=\"%s\"` to see the embeddings.", log_dir.replace("\\", "\\\\"))


if __name__ == '__main__':
    import argparse

    trim_embeddings()
    logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(filename)s::%(funcName)s\n%(message)s',
                        level=logging.INFO)

    this_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Prepare embeddings for visualization.")
    parser.add_argument('--embeddings_path', type=str,
                        default=os.path.join(this_dir, 'data', 'glove.6B', 'glove.6B.100d.txt'),
                        help="Path to the embeddings.")
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(this_dir, 'log_dir'),
                        help="Directory to save the embeddings to for TensorBoard.")

    args = parser.parse_args()

    prepare_embeddings_for_tensorboard(args.embeddings_path, args.log_dir)
#############################################################################################################################