import tensorflow as tf
from .base_score import BaseScorer


class TranSparseScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, mask, gamma, weight):
        super().__init__(head, relation, tail, mode)
        self.mask = mask
        self.gamma = gamma
        self.weight = weight

    def compute_score(self):
        p_head = tf.matmul(self.head, (self.mask * self.weight))
        p_head = tf.linalg.normalize(p_head, ord=2, axis=-1)[0]
        p_tail = tf.matmul(self.head, (self.mask * self.weight))
        p_tail = tf.linalg.normalize(p_tail, ord=2, axis=-1)[0]
        relation = tf.linalg.normalize(self.relation, ord=2, axis=-1)[0]

        score = p_head * relation - p_tail
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score
