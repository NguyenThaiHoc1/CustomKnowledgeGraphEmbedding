import tensorflow as tf
from .base_score import BaseScorer


class RotateCTScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode):
        super().__init__(head, relation, tail, mode)

    def compute_score(self):
        re_head, im_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        r_phase, re_b, im_b = tf.split(self.relation, num_or_size_splits=3, axis=2)
        re_tail, im_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)

        c_head = tf.complex(re_head, im_head)
        c_tail = tf.complex(re_tail, im_tail)
        c_b = tf.complex(re_b, im_b)
        c_r = tf.complex(
            tf.ones_like(r_phase) * tf.cos(r_phase),
            tf.ones_like(r_phase) * tf.sin(r_phase))

        score = (c_head - c_b) * c_r - (c_tail - c_b)
        score = self.gamma - tf.norm(score, ord=1, axis=-1)  # Default: norm 1
        return score
