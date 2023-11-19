import tensorflow as tf
import numpy as np
from .score_functions import (
    InterHTScorer, DistMultScorer, ComplEXScorer,
    RotatEScorer, RotateCTScorer, RotProScorer,
    STransEScorer, TranSScorer, TransDScorer, 
    TransEScorer, TripleREScorer, TranSparseScorer, TransRScorer
)
from .metrics import MeanReciprocalRank, MeanRank, HitsAt1


class TFKGEModel(tf.keras.Model):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False,
                 triple_relation_embedding=False, quora_relation_embedding=False,
                 pre_weights = None, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pi = tf.constant(np.pi, dtype=tf.float32)

        self.gamma = tf.Variable([gamma], trainable=False, dtype=tf.float32)

        self.embedding_range = tf.Variable([(self.gamma.numpy() + self.epsilon) / hidden_dim],
                                           trainable=False, dtype=tf.float32)

        initializer_range = (self.gamma.numpy() + self.epsilon) / hidden_dim
        initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)

        # enitty
        if double_entity_embedding:
            self.entity_dim = hidden_dim * 2
        else:
            self.entity_dim = hidden_dim

        # relation
        if double_relation_embedding:
            self.relation_dim = hidden_dim * 2
        elif triple_relation_embedding:
            self.relation_dim = hidden_dim * 3
        elif quora_relation_embedding:
            self.relation_dim = hidden_dim * 4
        else:
            self.relation_dim = hidden_dim

        if model_name in ['InterHT', 'TranS']:
            self.u = 1

        elif model_name in ['RotatE']:
            self.pi = tf.constant(np.pi, dtype=tf.float32)

        elif model_name in ['STransE']:
            self.W1 = tf.Variable(tf.zeros([self.entity_dim, self.entity_dim]), trainable=True)
            initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)
            self.W1.assign(initializer(self.W1.shape))

            self.W2 = tf.Variable(tf.zeros([self.entity_dim, self.entity_dim]), trainable=True)
            initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)
            self.W2.assign(initializer(self.W2.shape))

        elif model_name in ['TripleRE']:
            self.k = tf.sqrt(tf.cast(hidden_dim, tf.float32))

        elif model_name in ['TranSparse']:
            mask_list = []
            for i in range(nrelation):
                threshold = int(pre_weights[i] * 100)
                prob = tf.random.uniform(shape=[self.relation_dim, self.relation_dim], minval=1, maxval=100)
                mask_list.append(tf.where(prob >= threshold, 1.0, 0.0))
            self.mask = tf.Variable(tf.stack(mask_list, axis=0), trainable=False)

            self.W = tf.Variable(tf.zeros([nrelation, self.relation_dim, self.relation_dim]), trainable=True)
            self.W.assign(initializer(self.W.shape))
            
        elif model_name in ['TransR']:
            self.W = tf.Variable(tf.zeros([nrelation, self.relation_dim, self.relation_dim]), trainable=True)
            self.W.assign(initializer(self.W.shape))
            # self.mask = tf.Variable(tf.ones([nrelation, self.relation_dim, self.relation_dim]), trainable=False)

        self.entity_embedding = tf.Variable(tf.zeros([nentity, self.entity_dim]), trainable=True)
        self.relation_embedding = tf.Variable(tf.zeros([nrelation, self.relation_dim]), trainable=True)

        initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)
        self.entity_embedding.assign(initializer(self.entity_embedding.shape))

        initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)
        self.relation_embedding.assign(initializer(self.relation_embedding.shape))

        self.model_func = {
            'InterHT': self.InterHT,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotPro': self.RotPro,
            'RotateCT': self.RotateCT,
            'STransE': self.STransE,
            'TranS': self.TranS,
            'TransD': self.TransD,
            'TransE': self.TransE,
            'TripleRE': self.TripleRE,
            'TranSparse': self.TranSparse,
            'TransR': self.TransR,
        }

        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mrr_tracker = MeanReciprocalRank(name='mrr')
        self.mr_tracker = MeanRank(name='mr')
        self.hitsat1_tracker = HitsAt1(name='hitsat1')

    def positive_call(self, sample, training=True, **kwargs):
        sample, mode = sample

        def single_mode(sample, mode):
            positive_sample, negative_sample = sample

            head = tf.gather(self.entity_embedding, positive_sample[:, 0])
            head = tf.expand_dims(head, axis=1)

            relation = tf.gather(self.relation_embedding, positive_sample[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, positive_sample[:, 2])
            tail = tf.expand_dims(tail, axis=1)
            
            if self.model_name in ['TranSparse']:
                W = tf.gather(self.W, positive_sample[:, 1])
                mask = tf.gather(self.mask, positive_sample[:, 1])
                single_score = self.model_func[self.model_name](head, relation, tail, mode, W, mask)
            elif self.model_name in ['TransR']:
                W = tf.gather(self.W, positive_sample[:, 1])
                single_score = self.model_func[self.model_name](head, relation, tail, mode, W)
            else:
                single_score = self.model_func[self.model_name](head, relation, tail, mode)
                
            return single_score

        positive_score = single_mode(sample, mode)
        return positive_score

    def negative_call(self, sample, training=True, **kwargs):
        sample, mode = sample

        def head_batch_mode(sample, mode):
            tail_part, head_part = sample
            batch_size, negative_sample_size = tf.shape(head_part)[0], tf.shape(head_part)[1]

            head = tf.gather(self.entity_embedding, tf.reshape(head_part, [-1]))
            head = tf.reshape(head, [batch_size, negative_sample_size, -1])

            relation = tf.gather(self.relation_embedding, tail_part[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, tail_part[:, 2])
            tail = tf.expand_dims(tail, axis=1)

            if self.model_name in ['TranSparse']:
                W = tf.gather(self.W, tail_part[:, 1])
                mask = tf.gather(self.mask, tail_part[:, 1])
                negative_head_score = self.model_func[self.model_name](head, relation, tail, mode, W, mask)
            elif self.model_name in ['TransR']:
                W = tf.gather(self.W, tail_part[:, 1])
                negative_head_score = self.model_func[self.model_name](head, relation, tail, mode, W)
            else:
                negative_head_score = self.model_func[self.model_name](head, relation, tail, mode)
            return negative_head_score

        def tail_batch_mode(sample, mode):
            head_part, tail_part = sample
            batch_size, negative_sample_size = tf.shape(tail_part)[0], tf.shape(tail_part)[1]

            head = tf.gather(self.entity_embedding, head_part[:, 0])
            head = tf.expand_dims(head, axis=1)

            relation = tf.gather(self.relation_embedding, head_part[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, tf.reshape(tail_part, [-1]))
            tail = tf.reshape(tail, [batch_size, negative_sample_size, -1])

            if self.model_name in ['TranSparse']:
                W = tf.gather(self.W, head_part[:, 1])
                mask = tf.gather(self.mask, head_part[:, 1])
                negative_tail_score  = self.model_func[self.model_name](head, relation, tail, mode, W, mask)
            elif self.model_name in ['TransR']:
                W = tf.gather(self.W, head_part[:, 1])
                negative_tail_score  = self.model_func[self.model_name](head, relation, tail, mode, W)
            else:
                negative_tail_score = self.model_func[self.model_name](head, relation, tail, mode)
            return negative_tail_score

        head_score = head_batch_mode(sample, mode)
        tail_score = tail_batch_mode(sample, mode)
        negative_condition = tf.cond(tf.equal(mode, 0), lambda: 1.0, lambda: 0.0)
        return head_score * negative_condition + tail_score * (1 - negative_condition)

    def InterHT(self, head, relation, tail, mode):
        return InterHTScorer(head, relation, tail, mode, None, None, u=self.u, gamma=self.gamma).compute_score()

    def DistMult(self, head, relation, tail, mode):
        return DistMultScorer(head, relation, tail, mode, None, None).compute_score()

    def ComplEx(self, head, relation, tail, mode):
        return ComplEXScorer(head, relation, tail, mode, None, None).compute_score()

    def TransE(self, head, relation, tail, mode):
        return TransEScorer(head, relation, tail, mode, None, None, gamma=self.gamma).compute_score()

    def TransD(self, head, relation, tail, mode):
        return TransDScorer(head, relation, tail, mode, None, None).compute_score()

    def STransE(self, head, relation, tail, mode):
        return STransEScorer(head, relation, tail, mode, None, None,
                             w1=self.W1, w2=self.W2).compute_score()

    def TripleRE(self, head, relation, tail, mode):
        return TripleREScorer(head, relation, tail, mode, None, None,
                              k=self.k, gamma=self.gamma).compute_score()

    def TranS(self, head, relation, tail, mode):
        return TranSScorer(head, relation, tail, mode, None, None, u=self.u, gamma=self.gamma).compute_score()

    def RotatE(self, head, relation, tail, mode):
        return RotatEScorer(head, relation, tail, mode, None, None,
                            embedding_range=self.embedding_range,
                            pi=self.pi,
                            gamma=self.gamma).compute_score()

    def RotPro(self, head, relation, tail, mode):
        return RotProScorer(head, relation, tail, mode, None, None, embedding_range=self.embedding_range, pi=self.pi, gamma=self.gamma).compute_score()

    def RotateCT(self, head, relation, tail, mode):
        return RotateCTScorer(head, relation, tail, mode, None, None, embedding_range=self.embedding_range, pi=self.pi, gamma=self.gamma).compute_score()

    def TranSparse(self, head, relation, tail, mode, W, mask):
        return TranSparseScorer(head, relation, tail, mode, W, mask, gamma=self.gamma).compute_score()

    def TransR(self, head, relation, tail, mode, W):
        return TransRScorer(head, relation, tail, mode, W, None, gamma=self.gamma).compute_score()

    def train_step(self, data, **kwargs):
        positive_sample, negative_sample, subsampling_weight, mode = data

        with tf.GradientTape() as tape:
            negative_score = self.negative_call(((positive_sample, negative_sample), mode[0]), training=True)
            negative_score = tf.reduce_sum(
                tf.nn.softmax(negative_score * 1, axis=1) * tf.math.log_sigmoid(-negative_score), axis=1,
                keepdims=True
            )
            positive_score = self.positive_call(((positive_sample, negative_sample), 3), training=True)
            positive_score = tf.math.log_sigmoid(positive_score)
            positive_sample_loss = -tf.reduce_sum(subsampling_weight * positive_score) / tf.reduce_sum(
                subsampling_weight)
            negative_sample_loss = -tf.reduce_sum(subsampling_weight * negative_score) / tf.reduce_sum(
                subsampling_weight)
            loss = (positive_sample_loss + negative_sample_loss) / 2

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data, **kwargs):
        positive_sample, negative_sample, filter_bias, mode = data
        score = self.negative_call(((positive_sample, negative_sample), mode[0]), training=False)
        score += filter_bias
        argsort = tf.argsort(score, axis=1, direction='DESCENDING')
        positive_arg = tf.cond(tf.equal(mode[0], 0), lambda: positive_sample[:, 0], lambda: positive_sample[:, 2])
        positive_arg = tf.cast(positive_arg, dtype=tf.int32)
        rankings = tf.where(tf.equal(argsort, tf.expand_dims(positive_arg, axis=-1)))
        true_rankings = rankings[:, -1] + 1

        # Update the metrics.
        self.mrr_tracker.update_state(true_rankings)
        self.mr_tracker.update_state(true_rankings)
        self.hitsat1_tracker.update_state(true_rankings)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            "MRR": self.mrr_tracker.result(),
            "MR": self.mr_tracker.result(),
            "HIT@1": self.hitsat1_tracker.result()
        }
