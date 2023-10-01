import tensorflow as tf


class MeanReciprocalRank(tf.keras.metrics.Metric):
    def __init__(self, name="mean_reciprocal_rank", **kwargs):
        super(MeanReciprocalRank, self).__init__(name=name, **kwargs)
        self.total_rank = self.add_weight(name="total_rank", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, true_rankings, sample_weight=None):
        """
        Update the metric state.

        Parameters:
        - true_rankings: A tensor containing the true rankings.
        - sample_weight: Optional sample weights (not used in this metric).
        """
        self.total_rank.assign_add(tf.reduce_sum(1.0 / tf.cast(true_rankings, dtype=tf.float32)))
        self.count.assign_add(tf.cast(tf.shape(true_rankings)[0], dtype=tf.float32))

    def result(self):
        """
        Compute and return the mean reciprocal rank.
        """
        return self.total_rank / self.count

    def reset_state(self):
        """
        Reset the metric state.
        """
        self.total_rank.assign(0.0)
        self.count.assign(0.0)

# class MeanReciprocalRank(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super(MeanReciprocalRank, self).__init__(**kwargs)
#         self.result = self.add_weight('tp', initializer='zeros')
#
#     def update_state(self, true_rankings):
#         mrr = 1.0 / tf.cast(true_rankings, dtype=tf.float32)
#         self.result.assign_add(tf.reduce_sum(tf.cast(mrr, self.dtype)))
#
#     def reset_state(self):
#         self.result().assign(0)
#
#     def result(self):
#         return self.result
