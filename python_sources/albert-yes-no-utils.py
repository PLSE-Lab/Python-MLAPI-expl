import collections
import copy
import json
import numpy as np
import tensorflow.compat.v1 as tf
from scipy.special import softmax
from tqdm import tqdm


albert_yes_no_config = {
    "max_seq_length": 512
}


class InputFeatureCLS:
    """A single set of features of data with answer type."""

    def __init__(self,
                 example_id,
                 context_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 answer_type=0):
        self.example_id = example_id
        self.context_id = context_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answer_type = answer_type


def get_text(e, c: dict):
    """Returns the text in the example's document in the given token span."""
    tokens = []

    for i in range(c["start_token"], c["end_token"]):
        token = e["document_tokens"][i]
        token = token.replace(" ", "")
        if token.startswith("<") and token.endswith(">"):  # skip html tokens
            continue

        tokens.append(token)
    return " ".join(tokens)


def make_question(e: dict):
    """Add text and type of questions."""
    keyword_list = ["are", "is", "was", "were",
                    "can", "could", "will", "would", "should",
                    "did", "do", "does", "has", "had", "have"]

    input_text = e["question_text"]
    yes_no_question = ("true or false" in input_text) or (input_text.strip().split(" ")[0] in keyword_list)
    question = {"input_text": input_text,
                "yes_no_question": yes_no_question}

    return question


def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    # There is no annotation for test files
    # For the sake of the competition let's just use a flag here. It's so ugly though..

    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    try:
        e["document_tokens"] = e["document_text"].split(" ")
    except KeyError:
        e["document_tokens"] = [t["token"] for t in e["document_tokens"]]

    question = make_question(e)

    context_idxs = []
    context_list = []

    for idx, c in enumerate(e["long_answer_candidates"]):
        # for training, keep only annotated contexts for w questions, and all contexts for yes/no questions
        if not c["top_level"]:
            continue

        # Many contexts are empty once the HTML tags have been stripped, so we want to skip those.
        text = get_text(e, c)
        if not text.strip():
            continue

        # this runs both for normal questions and yes/no questions
        context = {"id": idx,
                   "text": text}
        context_idxs.append(idx)
        context_list.append(context)

    # Assemble example.
    example = {"id": e["example_id"],
               "question": question,
               "context_list": context_list}

    return example


def tokenize_by_sp(tokenizer, text):
    """Tokenizes text by sentencepiece. This should not contain special tokens"""
    tokens = tokenizer.tokenize(text.lower())

    return tokens


def convert_single_example(example: dict, albert_tokenizer):
    """Converts a single NqExample into a list of InputFeatures."""
    max_seq_length = albert_yes_no_config["max_seq_length"]

    features_yes_no = []

    # QUERY
    query_tokens_sp = ["[CLS]"]
    query_tokens_sp.extend(tokenize_by_sp(albert_tokenizer, example["question"]["input_text"]))
    query_tokens_sp.append("[SEP]")

    # CONTEXTS
    context_list = example["context_list"]          # keys = "id", "text"

    if example["question"]["yes_no_question"]:
        for context in context_list:
            answer_type = 0

            segment_ids = [0] * len(query_tokens_sp)

            text = context["text"]
            text_tokens = tokenize_by_sp(albert_tokenizer, text)

            # pad/truncate to max_seq_length_long and add to 2a
            text_tokens = text_tokens[:max_seq_length - len(query_tokens_sp) - 1]
            tokens = query_tokens_sp + text_tokens + ["[SEP]"]
            segment_ids += [1] * (len(text_tokens) + 1)
            input_ids = albert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids.extend(padding)
            input_mask.extend(padding)
            segment_ids.extend(padding)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            feature_yes_no = InputFeatureCLS(example_id=-1,
                                             context_id=context["id"],
                                             input_ids=input_ids,
                                             input_mask=input_mask,
                                             segment_ids=segment_ids,
                                             answer_type=answer_type)
            features_yes_no.append(feature_yes_no)

    return features_yes_no


def read_nq_examples(input_file):
    """Read a NQ json file into a list of NqExample."""
    examples = []

    with open(input_file, "r") as f:
        for line in f:
            examples.append(create_example_from_jsonl(line))

    return examples


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


class FeatureWriter:
    """Writes InputFeature to TF example file."""
    def __init__(self, filename, albert_tokenizer):
        self.albert_tokenizer = albert_tokenizer
        self._writer = tf.python_io.TFRecordWriter(filename)

    def file_based_convert_examples_to_features_span(self, examples):
        """Converts a list of NqExamples into InputFeatures, file-based"""
        example_to_feature_map = {}                 # dict of {example_id: [feature_idxs}}
        feature_to_example_map = []                 # list of (example_id, context_id)
        n_features = 0

        for i, example in enumerate(tqdm(examples)):
            example_index = int(example["id"])
            example_to_feature_map[example_index] = []
            features_yes_no = convert_single_example(example, self.albert_tokenizer)

            for feature in features_yes_no:
                feature.example_index = example_index
                feature.unique_id = feature.example_index + feature.context_id
                feature_to_example_map.append((feature.example_index, feature.context_id))

                features = collections.OrderedDict()
                features["example_ids"] = create_int_feature([feature.example_index])
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_int_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                self._writer.write(tf_example.SerializeToString())

                example_to_feature_map[example_index].append(n_features)
                n_features += 1

        self._writer.close()

        return n_features, example_to_feature_map, feature_to_example_map


default_pred_dict = {"example_id": None,
                     "long_answer": {"start_token": -1,
                                     "end_token": -1,
                                     "start_byte": -1,
                                     "end_byte": -1},
                     "long_answer_score": -10000,
                     "short_answers": [{"start_token": -1,
                                        "end_token": -1,
                                        "start_byte": -1,
                                        "end_byte": -1}],
                     "short_answers_score": -10000,
                     "yes_no_answer": "NONE",
                     "answer_type_probabilities": [],
                     "answer_type": 0
                     }


def compute_pred_dicts_cls(predict_file, results, example_to_feature_map, feature_to_example_map):
    # example_to_feature_map = {}  # dict of {example_id: [feature_idxs}}
    # feature_to_example_map = []  # list of (example_id, context_id)
    assert len(results) == len(feature_to_example_map)
    pred_dicts = []

    f = open(predict_file, "r")
    for line in f:
        e = json.loads(line.strip())
        example_id = int(e["example_id"])
        pred_dict = copy.deepcopy(default_pred_dict)
        pred_dict["example_id"] = example_id

        for feature_id in example_to_feature_map[example_id]:
            result = results[feature_id]
            probabilities = softmax(result)

            assert feature_to_example_map[feature_id][0] == example_id
            context_id = feature_to_example_map[feature_id][1]

            answer_type = int(np.argmax(probabilities))
            probability = float(np.max(probabilities))
            pred_dict["answer_type_probabilities"].append((context_id, answer_type, probability))

        for context_id, answer_type, probability in sorted(pred_dict["answer_type_probabilities"],
                                                           key=lambda tup: tup[2],
                                                           reverse=True):
            if answer_type == 0:
                continue
            else:
                c = e["long_answer_candidates"][context_id]
                start_token = c["start_token"]
                end_token = c["end_token"]

                pred_dict["long_answer_score"] = probability
                pred_dict["long_answer"]["start_token"] = start_token
                pred_dict["long_answer"]["end_token"] = end_token
                if answer_type == 1:
                    pred_dict["yes_no_answer"] = "YES"
                elif answer_type == 2:
                    pred_dict["yes_no_answer"] = "NO"
                break

        pred_dicts.append(pred_dict)

    f.close()

    return pred_dicts
