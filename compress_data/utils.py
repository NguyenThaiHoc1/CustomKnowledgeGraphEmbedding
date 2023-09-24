import tensorflow as tf


# WRITTING TFRECORD FILE
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(positive_sample, negative_sample, subsampling_weight, mode):
    feature = {
        "positive_sample": int_feature_list(positive_sample),
        "negative_sample": int_feature_list(negative_sample),
        "subsampling_weight": float_feature_list(subsampling_weight),
        "mode": int64_feature(mode),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# READING TFRECORD FILE & CONVERT SHAPE
def parse_tfrecord_fn(example):
    feature_description = {
        "positive_sample": tf.io.VarLenFeature(tf.int64),
        "negative_sample": tf.io.VarLenFeature(tf.int64),
        "subsampling_weight": tf.io.VarLenFeature(tf.float32),
        "mode": tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["positive_sample"] = tf.sparse.to_dense(example["positive_sample"])
    example["negative_sample"] = tf.sparse.to_dense(example["negative_sample"])
    example["subsampling_weight"] = tf.sparse.to_dense(example["subsampling_weight"])
    example["mode"] = tf.sparse.to_dense(example["mode"])
    return example


def reshape_function(example, batch_size):
    postive_sample = example["positive_sample"]
    negative_sample = example["negative_sample"]
    subsampling_weight = example["subsampling_weight"]
    mode = example["mode"]

    postive_sample = tf.reshape(postive_sample, [-1])
    negative_sample = tf.reshape(negative_sample, [-1])
    subsampling_weight = tf.reshape(subsampling_weight, [-1])
    mode = tf.reshape(mode, [1])

    return postive_sample, negative_sample, subsampling_weight, mode
