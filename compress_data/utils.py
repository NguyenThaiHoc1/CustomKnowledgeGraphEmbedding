import tensorflow as tf


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
        "mode": int_feature_list(mode),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
