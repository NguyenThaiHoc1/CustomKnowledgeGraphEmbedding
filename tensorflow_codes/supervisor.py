import tensorflow as tf
import time


class Trainer:
    def __init__(self, strategy, dataloader, model, optimizer, metrics):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.strategy = strategy

    @tf.function
    def train_step(self, data_iter):
        def train_step_fn(positive_sample, negative_sample, subsampling_weight, mode):
            with tf.GradientTape() as tape:
                negative_score = self.model(((positive_sample, negative_sample), mode[0]))
                positive_score = self.model(((positive_sample, negative_sample), 3))
                positive_sample_loss = -tf.reduce_sum(subsampling_weight * positive_score) / tf.reduce_sum(subsampling_weight)
                negative_sample_loss = -tf.reduce_sum(subsampling_weight * negative_score) / tf.reduce_sum(subsampling_weight)
                loss = (positive_sample_loss + negative_sample_loss) / 2

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.metrics.update_state(loss * self.strategy.num_replicas_in_sync)

        self.strategy.run(train_step_fn, next(data_iter))

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
                  'loss: {:0.4f}'.format(round(float(self.metrics.result()), 4)),
                  flush=True)
            epoch = step // steps_per_epoch
            epoch_steps = 0
            epoch_start_time = time.time()
            self.metrics.reset_states()
            if epoch >= epochs:
                break

        print("DONE")
