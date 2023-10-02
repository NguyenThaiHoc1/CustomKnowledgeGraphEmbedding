import tensorflow as tf
from .base_score import BaseScorer


class RotProScorer(BaseScorer):

    def __init__(self, head, relation, tail, mode, embedding_range, pi, gamma):
        super().__init__(head, relation, tail, mode)
        self.embedding_range = embedding_range
        self.pi = pi
        self.gamma = gamma

    def compute_score(self):
        re_head, im_head = tf.split(self.head, num_or_size_splits=2, axis=2)
        r_phase, proj_a, proj_b, proj_phase = tf.split(self.relation, num_or_size_splits=4, axis=2)
        re_tail, im_tail = tf.split(self.tail, num_or_size_splits=2, axis=2)
        r_phase = r_phase / (self.embedding_range / self.pi)

        re_r_phase = tf.ones_like(r_phase) * tf.cos(r_phase)
        im_r_phase = tf.ones_like(r_phase) * tf.sin(r_phase)

        re_proj_phase = tf.ones_like(proj_phase) * tf.cos(proj_phase)
        im_proj_phase = tf.ones_like(proj_phase) * tf.sin(proj_phase)

        def pr(re, im):
            return [re * ma + im * mb, re * mb + im * md]

        ma = tf.pow(re_proj_phase, 2) * proj_a + tf.pow(im_proj_phase, 2) * proj_b
        mb = re_proj_phase * im_proj_phase * (proj_b - proj_a)
        md = tf.pow(re_proj_phase, 2) * proj_b + tf.pow(im_proj_phase, 2) * proj_a

        tt_real, tt_img = pr(re_tail, im_tail)
        pr_re_head, pr_im_head = pr(re_head, im_head)
        hr_real = re_r_phase * pr_re_head + -im_r_phase * pr_im_head
        hr_img = im_r_phase * pr_re_head + re_r_phase * pr_im_head

        hr = tf.complex(hr_real, hr_img)
        tt = tf.complex(tt_real, tt_img)

        return self.gamma - tf.norm(hr - tt, ord=1, axis=-1)  # Default: 1 (like RotatE).
