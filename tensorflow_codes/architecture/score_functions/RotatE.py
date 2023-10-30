import tensorflow as tf
from .base_score import BaseScorer


class RotatEScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, W , mask, embedding_range, pi, gamma):
        super().__init__(head, relation, tail, mode, W, mask)
        self.embedding_range = embedding_range
        self.pi = pi
        self.gamma = gamma

    def compute_score(self):
        re_head, im_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        re_tail, im_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)
        phase_relation = self.relation / (self.embedding_range / self.pi)
        re_relation = tf.cos(phase_relation)
        im_relation = tf.sin(phase_relation)

        re_score_0 = re_relation * re_tail + im_relation * im_tail - re_head
        im_score_0 = re_relation * im_tail - im_relation * im_tail - im_head
        re_score_1 = re_head * re_relation - im_head * im_relation - re_tail
        im_score_1 = re_head * im_relation + im_head * re_relation - im_tail

        score_0 = tf.norm(tf.stack([re_score_0, im_score_0], axis=0), axis=0)
        score_1 = tf.norm(tf.stack([re_score_1, im_score_1], axis=0), axis=0)

        choice = tf.cond(tf.equal(self.mode, 0), lambda: 1.0, lambda: 0.0)

        score = score_0 * choice + score_1 * (1 - choice)
        score = self.gamma - tf.reduce_sum(score, axis=2)
        return score
