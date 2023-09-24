import tensorflow as tf
from .base_score import BaseScorer


class TransDScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode):
        super().__init__(head, relation, tail, mode)

    def compute_score(self):
        def _transfer(e, ep, rp):
            i = tf.eye(rp.shape[-1], e.shape[-1])
            rp = rp * tf.ones_like(tf.matmul(ep, i, transpose_b=True))
            m = tf.matmul(rp, ep, transpose_a=True)
            result = tf.matmul(e, m + i, transpose_b=True)
            return result

        head, p_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        relation, p_relation = tf.split(self.relation, num_or_size_splits=2, axis=2)
        tail, p_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)

        head_transfer = _transfer(head, p_head, p_relation)
        tail_transfer = _transfer(tail, p_tail, p_relation)
        return tf.norm(head_transfer + relation - tail_transfer, axis=-1, ord=2) ** 2
