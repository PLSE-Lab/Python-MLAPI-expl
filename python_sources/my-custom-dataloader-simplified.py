#!/usr/bin/env python
# coding: utf-8

# Another way of dataloader instead of using tf.Example and tf.Record. (I don't know the difference)
# This is just a sample and you can't run because of private utility scripts. sorry for inconvenient.

# In[ ]:


import sys
sys.path.append('../input/transformers')
import json
from typing import Tuple, List, Generator, Union, Mapping
from pathlib import Path
import numpy as np
import tensorflow as tf

from common_dataloader_base import BaseDataloader
from common_utils_config import Config, Mode
from common_preprocess_base import BasePreprocessor
from common_preprocess_nq_preprocess import NQPreprocessor
from common_entity_nq_entity import Sample, NQSample, Record


# In[ ]:


TrainInputType = Tuple[
    np.int64,    # unique_id
    np.int64,    # example_id
    np.ndarray,  # input_ids
    np.ndarray,  # input_mask
    np.ndarray,  # segment_ids
    np.ndarray,  # token_map
    np.ndarray,  # max_context_map
    np.int32,    # answer_type
    np.int32,    # short_start_token
    np.int32,    # short_end_token
    np.int32,    # long_start_token
    np.int32     # long_end_token
]

TestInputType = Tuple[
    np.int64,    # unique_id
    np.int64,    # example_id
    np.ndarray,  # input_ids
    np.ndarray,  # input_mask
    np.ndarray,  # segment_ids
    np.ndarray,  # token_map
    np.ndarray   # max_context_map
]


# In[ ]:


class NQDataloader(BaseDataloader):

    def __init__(self,
                 config: Config,
                 filepath: Path,
                 preprocessor: BasePreprocessor,
                 mode: Mode = Mode.TRAIN,
                 is_debug: bool = False) -> None:
        self.config = config
        self.is_debug = is_debug
        self.filepath = filepath
        self.mode = mode
        self.preprocessor = preprocessor

    def generator(self) -> Generator:
        '''Return Generator for tf.data.Dataset
        '''
        for line in open(self.filepath, 'r'):
            line = json.loads(line)
            record = Record(**line)

            samples, nq_sample = self.preprocess(record)
            for sample in samples:
                if self.is_debug:
                    yield sample, nq_sample, record
                elif self.mode != Mode.TRAIN:
                    yield self.to_input(sample, self.mode)
                else:
                    if sample.long_span is not None or sample.short_span is not None:
                        yield self.to_input(sample, self.mode)
                    else:
                        if np.random.rand() < self.config.downsample_rate:  # = 0.02
                            yield self.to_input(sample, self.mode)

    def preprocess(self, record: Record) -> Tuple[List[Sample], NQSample]:
        '''Apply preprocess by Preprocessor
        '''
        samples, nq_sample = self.preprocessor.preprocess(record)
        return samples, nq_sample

    def to_input(self,
                 sample: Sample,
                 mode: Mode) -> Union[TrainInputType, TestInputType]:
        '''Make model input
        '''
        unique_id = np.int64(sample.unique_id)
        example_id = np.int64(sample.example_id)
        input_ids = np.array(sample.input_ids, dtype=np.int32)
        input_mask = np.array(sample.input_mask, dtype=np.int32)
        segment_ids = np.array(sample.segment_ids, dtype=np.int32)
        token_map = [-1] * len(sample.input_ids)
        max_context_map = [-1] * len(sample.input_ids)
        for k, v in sample.wp_to_token_map.items():
            token_map[k] = v
        for k, v in sample.wp_token_to_max_context_map.items():
            max_context_map[k] = v
        if mode == Mode.TEST:
            return unique_id, example_id, input_ids, input_mask, segment_ids, np.array(token_map), np.array(max_context_map)
        else:
            answer_type = np.int32(sample.answer_type.value)
            short_start_token = np.int32(sample.short_span.start_token if sample.short_span else -1)
            short_end_token = np.int32(sample.short_span.end_token if sample.short_span else -1)
            long_start_token = np.int32(sample.long_span.start_token if sample.long_span else -1)
            long_end_token = np.int32(sample.long_span.end_token if sample.long_span else -1)
            return (unique_id, example_id, input_ids, input_mask, segment_ids, np.array(token_map), np.array(max_context_map),
                    answer_type, short_start_token, short_end_token, long_start_token, long_end_token)

    def to_dataset(self, mode: Mode) -> tf.data.Dataset:
        '''Return tf.data.Dataset instance for training
        '''
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=self.train_output_type if mode != Mode.TEST else self.test_output_type,
            output_shapes=self.train_output_shape if mode != Mode.TEST else self.test_output_shape
        )
        batch_size = self.config.batch_size if mode == Mode.TRAIN else self.config.test_batch_size
        dataset = dataset.batch(batch_size).prefetch(4)
        return dataset

    @property
    def train_output_type(self) -> Tuple:
        return (
            tf.int64,  # unique_id
            tf.int64,  # example_id
            tf.int32,  # input_ids
            tf.int32,  # input_mask
            tf.int32,  # segment_ids
            tf.int32,  # token_map
            tf.int32,  # max_context_map
            tf.int32,  # answer_type
            tf.int32,  # short_start_token
            tf.int32,  # short_end_token
            tf.int32,  # long_start_token
            tf.int32   # long_end_token
        )

    @property
    def train_output_shape(self) -> Tuple:
        return (
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape(())
        )

    @property
    def test_output_type(self) -> Tuple:
        return (
            tf.int64,  # unique_id
            tf.int64,  # example_id
            tf.int32,  # input_ids
            tf.int32,  # input_mask
            tf.int32,  # segment_ids
            tf.int32,  # token_map
            tf.int32,  # max_context_map
        )

    @property
    def test_output_shape(self) -> Tuple:
        return (
            tf.TensorShape(()),
            tf.TensorShape(()),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
            tf.TensorShape([self.config.max_seq_length]),
        )


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


from transformers import BertTokenizer


# In[ ]:


dummy_config = Config()
tokenizer = BertTokenizer(vocab_file='../input/tf2nq-vocab/vocab-nq.txt')
dummy_preprocessor = NQPreprocessor(dummy_config, tokenizer, mode=Mode.TEST)
filepath = Path('../input/tensorflow2-question-answering/simplified-nq-test.jsonl')
dataloader = NQDataloader(dummy_config, filepath, dummy_preprocessor, mode=Mode.TEST)


# get generator by `generator()` and convert generator into tf.data.Dataset by `to_dataset()`

# In[ ]:


get_ipython().run_cell_magic('time', '', 'generator = dataloader.generator()\n\ncount = 0\nfor _ in generator:\n    count += 1\nprint(count)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataset = dataloader.to_dataset(mode=Mode.TEST)\n\ncount = 0\nfor batch in dataset:\n    count += 1\nprint(count)')


# In[ ]:


print(batch)

