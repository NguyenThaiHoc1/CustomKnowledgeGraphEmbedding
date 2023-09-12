import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from compress_data.utils import create_example

from dataloader import DataLoader
from dataloader import TrainDataGenerator, TestDataGenerator
from dataloader import DataGenerator2Dataset, TestDataGenerator2Dataset

from dataloader_raw.regular_data import RegularDataRaw


def train_create_dataloader(input_dir, negative_sample_size, batch_size):
    dataraw_regular = RegularDataRaw(input_dir=input_dir)
    train_triples, valid_triples, test_triples, entity2id, relation2id = dataraw_regular.read()
    nentity = len(entity2id)
    nrelation = len(relation2id)

    # train
    train_generator_head = TrainDataGenerator(
        train_triples, nentity, nrelation, negative_sample_size, 0,  # 'head-batch'
    )

    train_generator_tail = TrainDataGenerator(
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


def test_create_dataloader(input_dir, batch_size):
    dataraw_regular = RegularDataRaw(input_dir=input_dir)
    train_triples, valid_triples, test_triples, entity2id, relation2id = dataraw_regular.read()
    nentity = len(entity2id)
    nrelation = len(relation2id)

    all_true_triples = train_triples + valid_triples + test_triples

    test_generator_head = TestDataGenerator(
        test_triples, all_true_triples, nentity, nrelation, 0
    )

    test_generator_tail = TestDataGenerator(
        test_triples, all_true_triples, nentity, nrelation, 1
    )

    test_dataset_head, test_length_head = TestDataGenerator2Dataset().convert(data_generator=test_generator_head)
    test_dataset_tail, test_length_tail = TestDataGenerator2Dataset().convert(data_generator=test_generator_tail)

    test_dataloader_head = DataLoader(test_dataset_head).gen_dataset(
        batch_size=batch_size, is_training=False, shuffle=False,
        input_pipeline_context=None, preprocess=None,
        drop_remainder=False
    )

    test_dataloader_tail = DataLoader(test_dataset_tail).gen_dataset(
        batch_size=batch_size, is_training=False, shuffle=False,
        input_pipeline_context=None, preprocess=None,
        drop_remainder=False
    )

    test_combined_dataset = tf.data.Dataset.sample_from_datasets(
        [test_dataloader_head, test_dataloader_tail],
        weights=[0.5, 0.5]
    )

    return test_combined_dataset, nrelation, nentity


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
    parser.add_argument("--split_number", type=int, default=10)
    parser.add_argument("--show_information", action='store_true')
    parser.add_argument("--mode", type=str, required=True, help="Train or Test")
    args = parser.parse_args()
    return args


def run(args):
    assert args.mode in ["train", "test"], 'Mode has just only "train" and "test".'

    if args.mode == "train":
        dataloader, nrelation, nentity = train_create_dataloader(input_dir=args.input_dir,
                                                                 negative_sample_size=args.negative_sample_size,
                                                                 batch_size=args.batch_size)

    else:
        dataloader, nrelation, nentity = test_create_dataloader(input_dir=args.input_dir,
                                                                batch_size=args.batch_size)

    if args.output_dir is not None:
        write_file_tfrecords(dataloader, args.output_dir, args.batch_size, split_number=args.split_number)

    if args.show_information:
        dataloader_length = len(list(dataloader))
        print(f"## {args.mode} - Information ###########")
        print(f"Len               :        {dataloader_length}")
        print(f"Number of sample  :        {dataloader_length * args.batch_size}")
        print(f"Number of relation:        {nrelation}")
        print(f"Number of entity  :        {nentity}")


if __name__ == '__main__':
    args = get_args()
    run(args)
