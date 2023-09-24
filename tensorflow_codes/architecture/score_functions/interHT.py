import tensorflow as tf
from .base_score import BaseScorer


class InterHTScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, u, gamma):
        super().__init__(head, relation, tail, mode)
        self.u = u
        self.gamma = gamma

    def compute_score(self):
        a_head, b_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        re_head, re_mid, re_tail = tf.split(self.relation, num_or_size_splits=3, axis=2)
        a_tail, b_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)

        e_h = tf.ones_like(b_head)
        e_t = tf.ones_like(b_tail)

        a_head = tf.linalg.normalize(a_head, ord=2, axis=-1)[0]
        a_tail = tf.linalg.normalize(a_tail, ord=2, axis=-1)[0]
        b_head = tf.linalg.normalize(b_head, ord=2, axis=-1)[0]
        b_tail = tf.linalg.normalize(b_tail, ord=2, axis=-1)[0]
        b_head = b_head + self.u * e_h
        b_tail = b_tail + self.u * e_t

        score = a_head * b_tail - a_tail * b_head + re_mid
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score
