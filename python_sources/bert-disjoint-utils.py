# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT-joint baseline for NQ v1.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import re
import enum
import numpy as np
import tensorflow.compat.v1 as tf

bert_disjoint_config = {
    "max_contexts": 48,
    "doc_stride": 128,
    "max_position": 50,
    "max_query_length": 64,
    "max_seq_length": 512,
    "n_best_size": 10,
    "max_answer_length": 25
}


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])
TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


class NqExample:
    """A single training/test example."""

    def __init__(self,
                 example_id,
                 qas_id,
                 questions,
                 doc_tokens,
                 doc_tokens_map=None,
                 answer=None,
                 start_position=None,
                 end_position=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 answer_text="",
                 answer_type=AnswerType.SHORT):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


def should_skip_context(e, idx):
    if not e["long_answer_candidates"][idx]["top_level"]:
        return True
    elif not get_candidate_text(e, idx).text.strip():
        # Skip empty contexts.
        return True
    else:
        return False


def get_text_span(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        token = t.replace(" ", "")

        if token.startswith("<") and token.endswith(">"):  # skip html tokens
            continue

        token_positions.append(i)
        tokens.append(token)

    return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
    """Converts a token index to the char offset within the candidate."""
    c = e["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(c["start_token"], token_idx):
        t = e["document_tokens"][i]
        token = t.replace(" ", "")

        if token.startswith("<") and token.endswith(">"):  # skip html tokens
            continue

        char_offset += len(token) + 1

    return char_offset


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        tf.logging.warning("Unknown candidate type found: %s", first_token)
        return "Other"


def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < bert_disjoint_config["max_position"]:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")

    # This returns an actual candidate.
    return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c


def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    e["document_tokens"] = e["document_text"].split(" ")
    add_candidate_types_and_positions(e)
    question = {"input_text": e["question_text"]}

    annotation, annotated_idx, annotated_sa = None, None, None
    answer = None

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (
        get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= bert_disjoint_config["max_contexts"]:
            break

    # Assemble example.
    example = {
        "id": str(e["example_id"]),
        "questions": [question],
        "answers": [answer],
        "has_correct_context": annotated_idx in context_idxs
    }

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" % (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map

    return example


def read_nq_entry(entry):
    """Converts a NQ entry into a list of NqExamples."""

    def is_whitespace(_c):
        return _c in " \t\r\n" or ord(_c) == 0x202F

    examples = []
    contexts_id = entry["id"]
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None

        questions.append(question_text)
        example = NqExample(
            example_id=int(contexts_id),
            qas_id=qas_id,
            questions=questions[:],
            doc_tokens=doc_tokens,
            doc_tokens_map=entry.get("contexts_map", None),
            answer=answer,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, output_fn):
    """Converts a list of NqExamples into InputFeatures."""
    num_spans_to_ids = collections.defaultdict(list)

    for example in examples:
        example_index = example.example_id
        features = convert_single_example(example, tokenizer)
        num_spans_to_ids[len(features)].append(example.qas_id)

        for feature in features:
            feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.doc_span_index
            output_fn(feature)

    return num_spans_to_ids


def convert_single_example(example, tokenizer):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
    # tokenized word tokens in the contexts. The word tokens might themselves
    # correspond to word tokens in a larger document, with the mapping given
    # by `doc_tokens_map`.
    if example.doc_tokens_map:
        tok_to_orig_index = [
            example.doc_tokens_map[index] for index in tok_to_orig_index
        ]

    # QUERY
    query_tokens = ["[Q]"]
    query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
    if len(query_tokens) > bert_disjoint_config["max_query_length"]:
        query_tokens = query_tokens[-bert_disjoint_config["max_query_length"]:]

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = bert_disjoint_config["max_seq_length"] - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, bert_disjoint_config["doc_stride"])

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (bert_disjoint_config["max_seq_length"] - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == bert_disjoint_config["max_seq_length"]
        assert len(input_mask) == bert_disjoint_config["max_seq_length"]
        assert len(segment_ids) == bert_disjoint_config["max_seq_length"]

        start_position = None
        end_position = None
        answer_text = ""

        feature = InputFeatures(
            unique_id=-1,
            example_index=-1,
            doc_span_index=doc_span_index,
            token_to_orig_map=token_to_orig_map,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text)

        features.append(feature)

    return features


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens


# prediction utils

class CreateTFExampleFn:
    """Functor for creating NQ tf.Examples."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, example):
        """Coverts an NQ example in a list of serialized tf examples."""
        nq_examples = read_nq_entry(example)
        input_features = []
        for nq_example in nq_examples:
            input_features.extend(convert_single_example(nq_example, self.tokenizer))

        for input_feature in input_features:
            input_feature.example_index = int(example["id"])
            input_feature.unique_id = input_feature.example_index + input_feature.doc_span_index

            def create_int_feature(values):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

            features = collections.OrderedDict()
            features["unique_ids"] = create_int_feature([input_feature.unique_id])
            features["input_ids"] = create_int_feature(input_feature.input_ids)
            features["input_mask"] = create_int_feature(input_feature.input_mask)
            features["segment_ids"] = create_int_feature(input_feature.segment_ids)

            token_map = [-1] * len(input_feature.input_ids)
            for k, v in input_feature.token_to_orig_map.items():
                token_map[k] = v
            features["token_map"] = create_int_feature(token_map)

            yield tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def read_nq_examples(input_file):
    """Read a NQ json file into a list of NqExample."""
    input_paths = tf.gfile.Glob(input_file)
    input_data = []

    for path in input_paths:
        tf.logging.info("Reading: %s", path)
        with tf.gfile.Open(path, "r") as input_file:
            for line in input_file:
                input_data.append(create_example_from_jsonl(line))

    examples = []
    for entry in input_data:
        examples.extend(read_nq_entry(entry))
    return examples


class FeatureWriter:
    """Writes InputFeature to TF example file."""
    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            _feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return _feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        token_map = [-1] * len(feature.input_ids)
        for k, v in feature.token_to_orig_map.items():
            token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


class EvalExample:
    """Eval data available for a single example."""
    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}


class ScoreSummary:
    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None


def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    # with gzip.GzipFile(fileobj=tf.gfile.Open(input_path)) as input_file:
    with open(input_path, "r") as input_file:
        tf.logging.info("Reading examples from: %s", input_path)
        for line in input_file:
            e = json.loads(line)
            candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    return candidates_dict


def read_candidates(input_pattern):
    """Read candidates with real multiple processes."""
    input_paths = tf.gfile.Glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
        final_dict.update(read_candidates_from_one_split(input_path))
    return final_dict


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = bert_disjoint_config["n_best_size"]
    max_answer_length = bert_disjoint_config["max_answer_length"]

    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = example.features[unique_id]["token_map"].int64_list.value

        raw_scores_start = result["start_logits"]
        raw_scores_end = result["end_logits"]

        start_indexes = get_best_indexes(raw_scores_start, n_best_size)
        end_indexes = get_best_indexes(raw_scores_end, n_best_size)
        for start_index in start_indexes:
            if token_map[start_index] == -1:
                continue
            for end_index in end_indexes:
                if end_index < start_index:
                    continue
                if token_map[end_index] == -1:
                    continue
                length = end_index - start_index + 1

                if length > max_answer_length:
                    answer_type = "long"
                else:
                    answer_type = "short"
                summary = ScoreSummary()
                summary.short_span_score = raw_scores_start[start_index] + raw_scores_end[end_index]
                summary.cls_token_score = raw_scores_start[0] + raw_scores_end[0]
                start_span = token_map[start_index]
                end_span = token_map[end_index] + 1

                # Span logits minus the cls logits seems to be close to the best.
                score = summary.short_span_score - summary.cls_token_score
                predictions.append((score, summary, start_span, end_span, answer_type))

    # Default empty prediction.
    score = -10000.0
    short_span = Span(-1, -1)
    long_span = Span(-1, -1)
    summary = ScoreSummary()

    if predictions:
        all_predictions = sorted(predictions, key=lambda tup: tup[0], reverse=True)
        score, summary, start_span, end_span, answer_type = all_predictions[0]

        if True:
            for c in example.candidates:
                if c["top_level"] and c["start_token"] <= start_span and c["end_token"] >= end_span:
                    long_span = Span(c["start_token"], c["end_token"])

                    if answer_type == "short":
                        short_span = Span(start_span, end_span)
                    break

    summary.predicted_label = {
        "example_id": example.example_id,
        "long_answer": {
            "start_token": long_span.start_token_idx,
            "end_token": long_span.end_token_idx,
            "start_byte": -1,
            "end_byte": -1
        },
        "long_answer_score": score,
        "short_answers": [{
            "start_token": short_span.start_token_idx,
            "end_token": short_span.end_token_idx,
            "start_byte": -1,
            "end_byte": -1
        }],
        "short_answers_score": score,
        "yes_no_answer": "NONE"
    }

    return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res["unique_id"] + 1), res) for res in raw_results]

    # Cast example id to int32 for each example, similarly to the raw results.
    sess = tf.Session()
    all_candidates = candidates_dict.items()
    example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates])).eval(session=sess)
    examples_by_id = list(zip(example_ids, all_candidates))

    # Cast unique_id also to int32 for features.
    feature_ids = []
    features = []
    for f in dev_features:
        feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
        features.append(f.features.feature)
    feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
    features_by_id = list(zip(feature_ids, features))

    # Join example with features and raw results.
    examples = []
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id, key=lambda tup: tup[0])
    for idx, datum in merged:
        if isinstance(datum, tuple):
            examples.append(EvalExample(datum[0], datum[1]))
        elif "token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    # Construct prediction objects.
    tf.logging.info("Computing predictions...")
    summary_dict = {}
    nq_pred_dict = {}
    for e in examples:
        summary = compute_predictions(e)
        summary_dict[e.example_id] = summary
        nq_pred_dict[e.example_id] = summary.predicted_label
        if len(nq_pred_dict) % 100 == 0:
            tf.logging.info("Examples processed: %d", len(nq_pred_dict))
    tf.logging.info("Done computing predictions.")

    return nq_pred_dict
