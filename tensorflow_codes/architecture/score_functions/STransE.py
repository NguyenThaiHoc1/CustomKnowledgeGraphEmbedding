import tensorflow as tf
from .base_score import BaseScorer


class STransEScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, w1, w2):
        super().__init__(head, relation, tail, mode)
        self.w1 = w1
        self.w2 = w2

    def compute_score(self):
        dis = tf.matmul(self.head, self.w1) + self.relation - tf.matmul(self.tail, self.w2)
        return self.gamma - tf.norm(dis, ord=1, axis=2)  # L1, L2
