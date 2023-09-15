import tensorflow as tf
import time
from tensorflow_codes.model import TFKGEModel


class Trainer:
    def __init__(self, model, optimizer, metrics, test_dataloader=None):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics

    # @tf.function()
    def train_step(self, sample):
      positive_sample, negative_sample, subsampling_weight, mode = sample
      with tf.GradientTape() as tape:
          negative_score = self.model.negative_call(((positive_sample, negative_sample), mode[0]), training=True)
          negative_score = tf.reduce_sum(
              tf.nn.softmax(negative_score * 1, axis=1) * tf.math.log_sigmoid(-negative_score), axis=1,
              keepdims=True
          )
          positive_score = self.model.positive_call(((positive_sample, negative_sample), 3), training=True)
          positive_score = tf.math.log_sigmoid(positive_score)
          positive_sample_loss = -tf.reduce_sum(subsampling_weight * positive_score) / tf.reduce_sum(
              subsampling_weight)
          negative_sample_loss = -tf.reduce_sum(subsampling_weight * negative_score) / tf.reduce_sum(
              subsampling_weight)
          loss = (positive_sample_loss + negative_sample_loss) / 2
          log = {
            'positive_sample_loss': positive_sample_loss,
            'negative_sample_loss': negative_sample_loss,
            'loss': loss
          }

      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

      self.metrics[0].update_state(loss )
      return {m.name: m.result() for m in self.metrics}




    @tf.function()
    def test_step(self, data_iter):
        def test_step_fn(positive_sample, negative_sample, filter_bias, mode):
            score = self.model.negative_call(((positive_sample, negative_sample), mode[0]), training=False)
            score += filter_bias
            argsort = tf.argsort(score, axis=1, direction='DESCENDING')
            positive_arg = tf.cond(tf.equal(mode[0], 0), lambda: positive_sample[:, 0], lambda: positive_sample[:, 2])
            positive_arg = tf.cast(positive_arg, dtype=tf.int32)
            rankings = tf.where(tf.equal(argsort, tf.expand_dims(positive_arg, axis=-1)))
            true_rankings = rankings[:, -1] + 1

            # Calculate evaluation metrics
            mrr = 1.0 / tf.cast(true_rankings, dtype=tf.float32)
            mr = tf.cast(true_rankings, dtype=tf.float32)
            hits_at_1 = tf.where(true_rankings <= 1, 1.0, 0.0)
            hits_at_3 = tf.where(true_rankings <= 3, 1.0, 0.0)
            hits_at_10 = tf.where(true_rankings <= 10, 1.0, 0.0)

            self.metrics["MRR"].update_state(mrr)
            self.metrics["MR"].update_state(mr)
            self.metrics["HITS_AT_1"].update_state(hits_at_1)
            self.metrics["HITS_AT_3"].update_state(hits_at_3)
            self.metrics["HITS_AT_10"].update_state(hits_at_10)

        self.strategy.run(test_step_fn, next(data_iter))



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

def getTFTrainer():
    raw_dataset = tf.data.TFRecordDataset('gs://hien7613storage2/datasets/KGE/wn18rr.tfrec')
    # train
    # if args.multiple_files:
    #     filenames = args.input_path
    #     print(f"Test {args.input_path} is a file")
    # else:
    #     filenames = tf.io.gfile.glob(os.path.join(args.input_path, "*.tfrec"))
    #     print(f"Train List files: \n {filenames}")
    batch_size = 16
    filenames = "gs://hien7613storage2/datasets/KGE/wn18rr.tfrec"
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    parsed_dataset = parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=batch_size))
    parsed_dataset = parsed_dataset.repeat()
    # test_path = 
    # test_filenames = tf.io.gfile.glob(os.path.join(test_path, "*.tfrec"))
    # print(f"Test List files: \n {test_filenames}")
    # test_raw_dataset = tf.data.TFRecordDataset(test_filenames)
    # test_parsed_dataset = test_raw_dataset.map(parse_tfrecord_fn)
    # test_parsed_dataset = test_parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=4))
    # test_parsed_dataset = test_parsed_dataset.repeat()


    kge_model = TFKGEModel(
            model_name="RotatE",
            nentity=40943,
            nrelation=11,
            hidden_dim=1000,
            gamma=24.0,
            double_entity_embedding=True,
        )
    optimizer = tf.keras.optimizers.Adam(0.1)
    training_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    trainer = Trainer(
        model=kge_model,
        optimizer=optimizer,
        metrics=[training_loss]
    )
    return trainer, kge_model, optimizer, parsed_dataset
