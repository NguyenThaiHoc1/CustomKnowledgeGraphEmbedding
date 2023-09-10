import tensorflow as tf
import os


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

    postive_sample = tf.reshape(postive_sample, [batch_size, -1])
    negative_sample = tf.reshape(negative_sample, [batch_size, -1])
    subsampling_weight = tf.reshape(subsampling_weight, [batch_size, -1])
    mode = tf.reshape(mode, [batch_size, ])

    return postive_sample, negative_sample, subsampling_weight, mode


folder_path = './split_data/wn18rr'

file_list = os.listdir(folder_path)

tfrec_files = [os.path.join(folder_path, file) for file in file_list if file.endswith(".tfrec")]

print(
    tfrec_files
)
raw_dataset = tf.data.TFRecordDataset(tfrec_files)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
parsed_dataset = parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=1024))

count_split = 0
for features in parsed_dataset:
    data = features
    a, b, c, d = data
    # print(a.shape, b.shape, c.shape, d.shape)
    count_split += 1

print(count_split)
