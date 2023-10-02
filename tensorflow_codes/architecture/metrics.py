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


class MeanRank(tf.keras.metrics.Metric):
    def __init__(self, name="mean_rank", dtype=tf.float32, **kwargs):
        super(MeanRank, self).__init__(name=name, dtype=dtype, **kwargs)
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

        # valid_mask = tf.not_equal(true_rankings, 0)
        # valid_true_rankings = tf.boolean_mask(true_rankings, valid_mask)
        #
        # self.total_rank.assign_add(tf.reduce_sum(valid_true_rankings))
        # self.count.assign_add(tf.cast(tf.shape(valid_true_rankings)[0], dtype=self.dtype))

    def result(self):
        """
        Compute and return the mean rank.
        """
        return tf.math.divide_no_nan(self.total_rank, self.count)

    def reset_state(self):
        """
        Reset the metric state.
        """
        self.total_rank.assign(0.0)
        self.count.assign(0.0)


class HitsAt1(tf.keras.metrics.Metric):
    def __init__(self, name="hits_at_1", dtype=tf.float32, **kwargs):
        super(HitsAt1, self).__init__(name=name, dtype=dtype, **kwargs)
        self.hits = self.add_weight(name="hits", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, true_rankings, sample_weight=None):
        """
        Update the metric state.

        Parameters:
        - true_rankings: A tensor containing the true rankings.
        - sample_weight: Optional sample weights (not used in this metric).
        """
        valid_mask = tf.not_equal(true_rankings, 0)
        valid_true_rankings = tf.boolean_mask(true_rankings, valid_mask)

        # Check if any of the valid true rankings are equal to 1
        hits_1 = tf.reduce_any(tf.equal(valid_true_rankings, 1))

        self.hits.assign_add(tf.cast(hits_1, dtype=self.dtype))
        self.total_samples.assign_add(1.0)

    def result(self):
        """
        Compute and return the Hits@1 metric.
        """
        return tf.math.divide_no_nan(self.hits, self.total_samples)

    def reset_state(self):
        """
        Reset the metric state.
        """
        self.hits.assign(0.0)
        self.total_samples.assign(0.0)
