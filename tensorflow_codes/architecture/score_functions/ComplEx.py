import tensorflow as tf
from .base_score import BaseScorer


class ComplEXScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode):
        super().__init__(head, relation, tail, mode)

    def compute_score(self):
        re_head, im_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        re_relation, im_relation = tf.split(self.relation, num_or_size_splits=2, axis=2)
        re_tail, im_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)

        re_score_0 = re_relation * re_tail + im_relation * im_tail
        im_score_0 = re_relation * im_tail - im_relation * re_tail
        score_0 = re_head * re_score_0 + im_head * im_score_0

        re_score_1 = re_head * re_relation - im_head * im_relation
        im_score_1 = re_head * im_relation + im_head * re_relation
        score_1 = re_score_1 * re_tail + im_score_1 * im_tail

        choice = tf.cond(tf.equal(self.mode, 0), lambda: 1.0, lambda: 0.0)
        score = score_0 * choice + score_1 * (1 - choice)

        score = tf.reduce_sum(score, axis=2)
        return score
