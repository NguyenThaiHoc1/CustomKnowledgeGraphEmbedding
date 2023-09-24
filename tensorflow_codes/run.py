import tensorflow as tf
import os
import argparse
from architecture.model import TFKGEModel
from supervisor import Trainer


def check_device():
    try:  # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:  # detect GPUs
        # strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy


def args_parser():
    parser = argparse.ArgumentParser(description="Training ...")
    parser.add_argument("-ip", "--input_path", required=True, type=str)
    parser.add_argument("-bz", "--batch_size", required=True, type=int)
    parser.add_argument("--test_path", required=False, type=str)

    parser.add_argument("-sf", "--score_functions", required=True, type=str)
    parser.add_argument("--nentity", required=True, type=int)
    parser.add_argument("--nrelation", required=True, type=int)
    parser.add_argument("--hidden_dim", required=True, type=int)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--epochs", required=False, type=int, default=1)
    parser.add_argument("--steps_per_epoch", required=False, type=int, default=1000)
    parser.add_argument("-de", "--double_entity_embedding", action='store_true')
    parser.add_argument("-dr", "--double_relation_embedding", action='store_true')
    parser.add_argument("-tr", "--triple_relation_embedding", action='store_true')
    parser.add_argument("--multiple_files", action='store_true')

    args = parser.parse_args()
    return args


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


def loading_data(head_dataloader_path, tail_dataloader_path, bz, do_training=True):
    try:
        # loading head file
        head_raw_dataset = tf.data.TFRecordDataset(head_dataloader_path)
        head_parsed_dataset = head_raw_dataset.map(parse_tfrecord_fn)
        head_parsed_dataset = head_parsed_dataset.map(
            lambda inputs: reshape_function(inputs, batch_size=args.batch_size))
        head_parsed_dataset = head_parsed_dataset.batch(bz)

        # loading tail file
        tail_raw_dataset = tf.data.TFRecordDataset(tail_dataloader_path)
        tail_parsed_dataset = tail_raw_dataset.map(parse_tfrecord_fn)
        tail_parsed_dataset = tail_parsed_dataset.map(
            lambda inputs: reshape_function(inputs, batch_size=args.batch_size))
        tail_parsed_dataset = tail_parsed_dataset.batch(bz)

        combined_dataset = tf.data.Dataset.sample_from_datasets(
            [head_parsed_dataset, tail_parsed_dataset],
            weights=[0.5, 0.5]
        )

        if do_training:
            combined_dataset = combined_dataset.repeat()

        return combined_dataset
    except Exception as e:
        raise ValueError(f"func loading_data: {e}")


@tf.function
def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5.0
    LR_SUSTAIN_EPOCHS = 0.0
    LR_EXP_DECAY = .8

    if float(epoch) < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * float(epoch) + LR_START
    elif float(epoch) < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (float(epoch) - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


def run(strategy, args):
    # reading data ...
    # train
    if not args.multiple_files:
        filenames_head = args.input_path
        filenames_tail = args.input_path
        print(f"Test {args.input_path} is a file")
    else:
        filenames_head = tf.io.gfile.glob(os.path.join(args.input_path, "*-head.tfrec"))
        filenames_tail = tf.io.gfile.glob(os.path.join(args.input_path, "*-tail.tfrec"))
        print(f"Head - Train List files: \n {filenames_head}")
        print(f"Tail - Train List files: \n {filenames_tail}")
    print("====" * 6)

    # test
    if args.test_path is not None:
        test_filenames_head = tf.io.gfile.glob(os.path.join(args.test_path, "*-head.tfrec"))
        test_filenames_tail = tf.io.gfile.glob(os.path.join(args.test_path, "*-tail.tfrec"))
        test_dataloader = loading_data(test_filenames_head, test_filenames_tail, bz=4, do_training=True)
        print(f"Head - Test List files: \n {test_filenames_head}")
        print(f"Tail - Test List files: \n {test_filenames_tail}")
    else:
        test_dataloader = None
    print("====" * 6)

    train_dataloader = loading_data(filenames_head, filenames_tail, bz=args.batch_size, do_training=True)
    print("1. Data loading complete.")

    assert args.score_functions not in ["InterHT", "DistMult"], f"{args.score_functions} is not implemented."
    print("2. Score function check complete.")

    with strategy.scope():
        kge_model = TFKGEModel(
            model_name=args.score_functions,
            nentity=args.nentity,
            nrelation=args.nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding,
            triple_relation_embedding=args.triple_relation_embedding
        )

        class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __call__(self, step):
                return lrfn(epoch=step // args.steps_per_epoch)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule())

        # metrics
        list_metrics = {
            "train_loss": tf.keras.metrics.Mean('training_loss', dtype=tf.float32),
            "MRR": tf.keras.metrics.Mean('mrr_evaluate', dtype=tf.float32),
            "MR": tf.keras.metrics.Mean('mr_evaluate', dtype=tf.float32),
            "HITS_AT_1": tf.keras.metrics.Mean('hit1_evaluate', dtype=tf.float32),
            "HITS_AT_3": tf.keras.metrics.Mean('hit3_evaluate', dtype=tf.float32),
            "HITS_AT_10": tf.keras.metrics.Mean('hit10_evaluate', dtype=tf.float32)
        }

        # supervisor
        trainer = Trainer(
            strategy=strategy,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=kge_model,
            optimizer=optimizer,
            metrics=list_metrics
        )
        print("3. Starting training...")
        trainer.training(
            steps_per_tpu_call=99,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch
        )
    print("4. Training complete.")


if __name__ == '__main__':
    strategy = check_device()
    args = args_parser()
    run(strategy, args)
