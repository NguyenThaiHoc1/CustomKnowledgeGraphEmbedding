import tensorflow as tf
from .base_score import BaseScorer


class TransRScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, W, mask, gamma):
        super().__init__(head, relation, tail, mode, W, mask)
        self.gamma = gamma

    def compute_score(self):
        p_head = tf.matmul(self.head, self.W)
        # p_head = tf.linalg.normalize(p_head, ord=2, axis=-1)[0]
        
        p_tail = tf.matmul(self.tail, self.W)
        # p_tail = tf.linalg.normalize(p_tail, ord=2, axis=-1)[0]
        
        relation = self.relation
        # relation = tf.linalg.normalize(self.relation, ord=2, axis=-1)[0]

        score = p_head + relation - p_tail
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score
