import tensorflow as tf
from .base_score import BaseScorer


class TripleREScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, W, mask, k, gamma):
        super().__init__(head, relation, tail, mode, W, mask)
        self.k = k
        self.gamma = gamma

    def compute_score(self):
        re_head, re_mid, re_tail = tf.split(self.relation, num_or_size_splits=3, axis=2)
        head = tf.math.l2_normalize(self.head, axis=-1)
        tail = tf.math.l2_normalize(self.tail, axis=-1)

        re_head = tf.math.l2_normalize(re_head, axis=-1)
        re_tail = tf.math.l2_normalize(re_tail, axis=-1)
        re_head = re_head * self.k
        re_tail = re_tail * self.k

        score = head * re_head - tail * re_tail + re_mid
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score
