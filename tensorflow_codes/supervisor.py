import tensorflow as tf
import time


class Trainer:
    def __init__(self, strategy, dataloader, model, optimizer, metrics):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.strategy = strategy

    @tf.function()
    def train_step(self, data_iter):
        def train_step_fn(positive_sample, negative_sample, subsampling_weight, mode):
            with tf.GradientTape() as tape:
                negative_score = self.model.negative_call(((positive_sample, negative_sample), mode[0]))
                positive_score = self.model.positive_call(((positive_sample, negative_sample), 3))
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

        def tetst_step_fn(positive_sample, negative_sample, filter_bias, mode):
            negative_score = self.model.negative_call(((positive_sample, negative_sample), mode[0]))
            negative_score += filter_bias
            argsort = tf.argsort(negative_score, axis=1, direction='DESCENDING')

            if mode == 'head-batch':
                positive_arg = tf.gather(positive_sample, indices=[0], axis=1)
            elif mode == 'tail-batch':
                positive_arg = tf.gather(positive_sample, indices=[2], axis=1)
            else:
                raise ValueError('mode %s not supported' % mode)

            for i in range(1024):
                # Notice that argsort is not ranking
                ranking = tf.where(tf.equal(argsort[i, :], positive_arg[i]))[:, 0]
                assert tf.shape(ranking)[0] == 1

                # Ranking + 1 is the true ranking used in evaluation metrics
                ranking = 1 + tf.cast(ranking[0], dtype=tf.float32)

                self.metrics["MRR"].update_state(1.0 / ranking)
                # logs.append({
                #     'MRR': 1.0 / ranking,
                #     'MR': ranking.numpy(),
                #     'HITS@1': 1.0 if ranking <= 1.0 else 0.0,
                #     'HITS@3': 1.0 if ranking <= 3.0 else 0.0,
                #     'HITS@10': 1.0 if ranking <= 10.0 else 0.0,
                # })

        self.strategy.run(tetst_step_fn, next(data_iter))

    def training(self, steps_per_tpu_call, epochs, steps_per_epoch):
        step = 0
        epoch = 0
        epoch_steps = 0
        epoch_start_time = time.time()
        iteration_data = iter(self.dataloader)
        while True:
            # run training step
            self.train_step(iteration_data)
            epoch_steps += steps_per_tpu_call
            step += steps_per_tpu_call
            print('=', end='', flush=True)

            # report metrics
            epoch_time = time.time() - epoch_start_time
            print('\nEPOCH {:d}/{:d}'.format(epoch + 1, epochs))
            print('time: {:0.1f}s'.format(epoch_time),
                  'loss: {:0.4f}'.format(round(float(self.metrics["train_loss"].result()), 4)),
                  flush=True)

            self.test_step(iter(self.dataloader))
            print('MRR: {:0.4f}'.format(round(float(self.metrics["MRR"].result()), 4)),
                  flush=True)

            epoch = step // steps_per_epoch
            epoch_steps = 0
            epoch_start_time = time.time()
            self.metrics["train_loss"].reset_states()
            if epoch >= epochs:
                break

        print("DONE")