import tensorflow as tf
import os
import argparse
from model import TFKGEModel
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

    parser.add_argument("-sf", "--score_function", required=True, type=str)
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

    postive_sample = tf.reshape(postive_sample, [batch_size, -1])
    negative_sample = tf.reshape(negative_sample, [batch_size, -1])
    subsampling_weight = tf.reshape(subsampling_weight, [batch_size, -1])
    mode = tf.reshape(mode, [batch_size, ])

    return postive_sample, negative_sample, subsampling_weight, mode


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
    if args.multiple_files:
        filenames = args.input_path
        print(f"Test {args.input_path} is a file")
    else:
        filenames = tf.io.gfile.glob(os.path.join(args.input_path, "*.tfrec"))
        print(f"Train List files: \n {filenames}")

    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    parsed_dataset = parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=args.batch_size))
    parsed_dataset = parsed_dataset.repeat()

    test_filenames = tf.io.gfile.glob(os.path.join(args.test_path, "*.tfrec"))
    print(f"Test List files: \n {test_filenames}")
    test_raw_dataset = tf.data.TFRecordDataset(test_filenames)
    test_parsed_dataset = test_raw_dataset.map(parse_tfrecord_fn)
    test_parsed_dataset = test_parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=4))
    test_parsed_dataset = test_parsed_dataset.repeat()

    with strategy.scope():
        kge_model = TFKGEModel(
            model_name=args.score_function,
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
            "HITS_AT_1": tf.keras.metrics.Mean('hit1_evaluate', dtype=tf.float32),
            "HITS_AT_3": tf.keras.metrics.Mean('hit3_evaluate', dtype=tf.float32),
            "HITS_AT_10": tf.keras.metrics.Mean('hit10_evaluate', dtype=tf.float32)
        }

        # supervisor
        trainer = Trainer(
            strategy=strategy,
            train_dataloader=parsed_dataset,
            test_dataloader=test_parsed_dataset,
            model=kge_model,
            optimizer=optimizer,
            metrics=list_metrics
        )
        trainer.training(
            steps_per_tpu_call=99,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch
        )


if __name__ == '__main__':
    strategy = check_device()
    args = args_parser()
    run(strategy, args)
