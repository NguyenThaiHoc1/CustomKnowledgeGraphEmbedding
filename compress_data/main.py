import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from compress_data.utils import create_example

from codes_tf.dataloader import DataLoader
from codes_tf.dataloader import DataGenerator
from codes_tf.dataloader import DataGenerator2Dataset


# def parse_tfrecord_fn(example):
#     feature_description = {
#         "image": tf.io.FixedLenFeature([], tf.string),
#         "path": tf.io.FixedLenFeature([], tf.string),
#         "area": tf.io.FixedLenFeature([], tf.float32),
#         "bbox": tf.io.VarLenFeature(tf.float32),
#         "category_id": tf.io.FixedLenFeature([], tf.int64),
#         "id": tf.io.FixedLenFeature([], tf.int64),
#         "image_id": tf.io.FixedLenFeature([], tf.int64),
#     }
#     example = tf.io.parse_single_example(example, feature_description)
#     example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
#     example["bbox"] = tf.sparse.to_dense(example["bbox"])
#     return example


def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_data(data_path):
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    return train_triples, valid_triples, test_triples, entity2id, relation2id


def create_dataloader(input_dir, negative_sample_size, batch_size):
    train_triples, valid_triples, test_triples, entity2id, relation2id = read_data(data_path=input_dir)
    nentity = len(entity2id)
    nrelation = len(relation2id)

    # train
    train_generator_head = DataGenerator(
        train_triples, nentity, nrelation, negative_sample_size, 0,  # 'head-batch'
    )

    train_generator_tail = DataGenerator(
        train_triples, nentity, nrelation, negative_sample_size, 1,  # 'tail-batch'
    )

    train_dataset_head, train_length_head = DataGenerator2Dataset().convert(data_generator=train_generator_head)
    train_dataset_tail, train_length_tail = DataGenerator2Dataset().convert(data_generator=train_generator_tail)

    train_dataloader_head = DataLoader(train_dataset_head).gen_dataset(
        batch_size=batch_size, is_training=True, shuffle=True,
        input_pipeline_context=None, preprocess=None,
        drop_remainder=False
    )

    train_dataloader_tail = DataLoader(train_dataset_tail).gen_dataset(
        batch_size=batch_size, is_training=True, shuffle=True,
        input_pipeline_context=None, preprocess=None,
        drop_remainder=False
    )

    combined_dataset = tf.data.Dataset.sample_from_datasets(
        [train_dataloader_head, train_dataloader_tail],
        weights=[0.5, 0.5]
    )

    # combined_dataset = combined_dataset.shuffle(2048)
    combined_dataset = combined_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return combined_dataset, nrelation, nentity


def write_file_tfrecords(
        dataloader, output_dir,
        batch_size, split_number
):
    dataset_name = os.path.basename(output_dir)
    len_dataloader = len(list(dataloader))
    total_sample_dataloader = len_dataloader
    each_sample_per_file = total_sample_dataloader // split_number
    iter_dataloader = iter(dataloader)
    print("3. Prepare ready.")
    for idx in tqdm(range(split_number)):
        path_output = os.path.join(output_dir, f"{dataset_name}-{idx}.tfrec")
        with tf.io.TFRecordWriter(path_output) as writer:
            for _ in range(each_sample_per_file):
                data = next(iter_dataloader)
                positive_sample, negative_sample, subsampling_weight, mode = data
                shape_batch_size = positive_sample.numpy().shape[0]
                if shape_batch_size != batch_size:
                    continue
                hstack_positive_sample = np.hstack(positive_sample.numpy())
                hstack_negative_sample = np.hstack(negative_sample.numpy())
                hstack_subsampling_weight = np.hstack(subsampling_weight.numpy())
                hstack_mode = np.hstack(mode.numpy())
                example = create_example(hstack_positive_sample,
                                         hstack_negative_sample,
                                         hstack_subsampling_weight,
                                         hstack_mode)
                writer.write(example.SerializeToString())


def get_args():
    parser = argparse.ArgumentParser(description="Compressing data with split tf_record file")
    parser.add_argument("-idr", "--input_dir", type=str, required=True, help="Input dicrection where contain data.")
    parser.add_argument("-odr", "--output_dir", type=str, required=False, help="Output dicrection where return output.")
    parser.add_argument("-bz", "--batch_size", type=int, required=True, help="Batch size which use in batch.")
    parser.add_argument("--negative_sample_size", type=int, default=256)
    args = parser.parse_args()
    return args


def run(args):
    split_number = 17
    print("1. Create Dataloader ...")
    dataloader, nrelation, nentity = create_dataloader(input_dir=args.input_dir,
                                                       negative_sample_size=args.negative_sample_size,
                                                       batch_size=args.batch_size)

    if args.output_dir is not None:
        print("2. Start writing ...")
        write_file_tfrecords(dataloader, args.output_dir, args.batch_size, split_number=split_number)

    dataloader_length = len(list(dataloader))
    print(f"## Information ###########")
    print(f"Len               :        {dataloader_length}")
    print(f"Number of sample  :        {dataloader_length * args.batch_size}")
    print(f"Number of relation:        {nrelation}")
    print(f"Number of entity  :        {nentity}")


if __name__ == '__main__':
    args = get_args()
    run(args)
