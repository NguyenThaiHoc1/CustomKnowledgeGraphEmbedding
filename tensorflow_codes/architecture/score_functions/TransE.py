import tensorflow as tf
from .base_score import BaseScorer


class TransEScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, W, mask, gamma):
        super().__init__(head, relation, tail, mode, W, mask)
        self.gamma = gamma

    def compute_score(self):
        score = self.head + self.relation - self.tail
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score
