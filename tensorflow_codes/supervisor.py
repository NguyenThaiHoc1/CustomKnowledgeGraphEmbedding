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
          print(log)

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

    # def training(self, steps_per_tpu_call, epochs, steps_per_epoch):
    #     step = 0
    #     epoch = 0
    #     epoch_steps = 0
    #     epoch_start_time = time.time()
    #     train_iteration_data = iter(self.train_dataloader)
    #     test_iteration_data = iter(self.test_dataloader)

    #     while True:
    #         # run training step
    #         self.train_step(train_iteration_data)
    #         epoch_steps += steps_per_tpu_call
    #         step += steps_per_tpu_call
    #         print('=', end='', flush=True)

    #         # report metrics
    #         epoch_time = time.time() - epoch_start_time
    #         print('\nEPOCH {:d}/{:d}'.format(epoch + 1, epochs))
    #         print('time: {:0.1f}s'.format(epoch_time),
    #               'loss: {:0.4f}'.format(round(float(self.metrics["train_loss"].result()), 4)),
    #               flush=True)

    #         # test
    #         self.test_step(test_iteration_data)
    #         print('Test step',
    #               'MRR: {:0.4f}'.format(round(float(self.metrics["MRR"].result()), 4)),
    #               'MR: {:0.4f}'.format(round(float(self.metrics["MR"].result()), 4)),
    #               'HITS_AT_1: {:0.4f}'.format(round(float(self.metrics["HITS_AT_1"].result()), 4)),
    #               'HITS_AT_3: {:0.4f}'.format(round(float(self.metrics["HITS_AT_3"].result()), 4)),
    #               'HITS_AT_10: {:0.4f}'.format(round(float(self.metrics["HITS_AT_10"].result()), 4)),
    #               flush=True)

    #         epoch = step // steps_per_epoch
    #         epoch_steps = 0
    #         epoch_start_time = time.time()
    #         self.metrics["train_loss"].reset_states()
    #         self.metrics["MRR"].reset_states()
    #         self.metrics["MR"].reset_states()
    #         self.metrics["HITS_AT_1"].reset_states()
    #         self.metrics["HITS_AT_3"].reset_states()
    #         self.metrics["HITS_AT_10"].reset_states()
    #         if epoch >= epochs:
    #             break

    #     print("DONE")



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

def getTFTrainer():
  raw_dataset = tf.data.TFRecordDataset('gs://hien7613storage2/datasets/KGE/wn18rr.tfrec')
  parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
  parsed_dataset = parsed_dataset.map(lambda inputs: reshape_function(inputs, batch_size=16))
  parsed_dataset = parsed_dataset

  kge_model = TFKGEModel(
          model_name="RotatE",
          nentity=40943,
          nrelation=11,
          hidden_dim=1000,
          gamma=24.0,
          double_entity_embedding=True,
      )
  optimizer = tf.keras.optimizers.Adam(0.)
  training_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

  trainer = Trainer(
      model=kge_model,
      optimizer=optimizer,
      metrics=[training_loss]
  )
  return trainer, parsed_dataset
