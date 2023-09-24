import tensorflow as tf
from .base_score import BaseScorer


class DistMultScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode):
        super().__init__(head, relation, tail, mode)

    def compute_score(self):
        score = self.head * self.relation * self.tail
        score = tf.reduce_sum(score, axis=2)
        return score
