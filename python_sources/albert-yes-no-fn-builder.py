import albert_yes_no_modeling as albert_modeling
import tensorflow.compat.v1 as tf


def create_albert_model(albert_config, input_ids, input_mask, segment_ids):
    albert_model = albert_modeling.AlbertModel(
        config=albert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )

    final_hidden = albert_model.get_pooled_output()

    answer_type_logits = tf.layers.dense(final_hidden,
                                         units=3,
                                         activation=None,
                                         use_bias=True,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         bias_initializer=tf.zeros_initializer())

    return answer_type_logits


def model_fn_builder(albert_config, init_checkpoint):
    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        example_ids = features["example_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        answer_type_logits = create_albert_model(
            albert_config=albert_config,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                albert_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        predictions = {"example_ids": example_ids,
                       "answer_type_logits": answer_type_logits}
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "example_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, _name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, _name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        # feature parsing
        d = d.map(lambda record: _decode_record(record, name_to_features),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        d = d.batch(batch_size=batch_size,
                    drop_remainder=drop_remainder)
        d = d.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # auto-tune at run time
        return d

    return input_fn
