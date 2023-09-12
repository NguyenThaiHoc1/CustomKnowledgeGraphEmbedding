import tensorflow as tf
import time


class Trainer:
    def __init__(self, strategy, train_dataloader, model, optimizer, metrics, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.strategy = strategy

    @tf.function()
    def train_step(self, data_iter):
        def train_step_fn(positive_sample, negative_sample, subsampling_weight, mode):
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

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.metrics["train_loss"].update_state(loss * self.strategy.num_replicas_in_sync)

        self.strategy.run(train_step_fn, next(data_iter))

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

        self.strategy.run(test_step_fn, next(data_iter))

    def training(self, steps_per_tpu_call, epochs, steps_per_epoch):
        step = 0
        epoch = 0
        epoch_steps = 0
        epoch_start_time = time.time()
        train_iteration_data = iter(self.train_dataloader)
        test_iteration_data = iter(self.test_dataloader)

        while True:
            # run training step
            self.train_step(train_iteration_data)
            epoch_steps += steps_per_tpu_call
            step += steps_per_tpu_call
            print('=', end='', flush=True)

            # report metrics
            epoch_time = time.time() - epoch_start_time
            print('\nEPOCH {:d}/{:d}'.format(epoch + 1, epochs))
            print('time: {:0.1f}s'.format(epoch_time),
                  'loss: {:0.4f}'.format(round(float(self.metrics["train_loss"].result()), 4)),
                  flush=True)

            # test
            self.test_step(test_iteration_data)
            print('Test step',
                  'MRR: {:0.4f}'.format(round(float(self.metrics["MRR"].result()), 4)),
                  flush=True)

            epoch = step // steps_per_epoch
            epoch_steps = 0
            epoch_start_time = time.time()
            self.metrics["train_loss"].reset_states()
            if epoch >= epochs:
                break

        print("DONE")
