import tensorflow as tf


def loss(model,
         inputs,
         regularization,
         negative_adversarial_sampling,
         uni_weight,
         adversarial_temperature):
    positive_sample, negative_sample, subsampling_weight, mode = inputs
    negative_score = model((positive_sample, negative_sample), mode=mode)

    if negative_adversarial_sampling:
        # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        negative_score = (tf.nn.softmax(negative_score * adversarial_temperature, axis=1).numpy()
                          * tf.math.log_sigmoid(-negative_score)).sum(axis=1)
    else:
        negative_score = tf.math.log_sigmoid(-negative_score).numpy().mean(axis=1)

    positive_score = model(positive_sample)

    positive_score = tf.math.log_sigmoid(positive_score).numpy().squeeze(axis=1)

    if uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss) / 2

    if regularization != 0.0:
        # Use L3 regularization for ComplEx and DistMult
        regularization = regularization * (
                tf.norm(model.entity_embedding, ord=3).numpy() ** 3 +
                tf.norm(tf.norm(model.relation_embedding, ord=3), ord=3).numpy() ** 3
        )
        loss = loss + regularization
        regularization_log = {'regularization': regularization}
    else:
        regularization_log = {}

    return loss


class TFKGEModel(tf.keras.Model):

    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False,
                 triple_relation_embedding=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = tf.Variable([gamma], trainable=False, dtype=tf.float32)

        self.embedding_range = tf.Variable([(self.gamma.numpy() + self.epsilon) / hidden_dim],
                                           trainable=False, dtype=tf.float32)

        if double_relation_embedding:
            self.relation_dim = hidden_dim * 2
        else:
            self.relation_dim = hidden_dim

        if double_entity_embedding:
            self.entity_dim = hidden_dim * 2
        else:
            self.entity_dim = hidden_dim

        if triple_relation_embedding:
            self.relation_dim = hidden_dim * 3
        else:
            self.relation_dim = hidden_dim

        if model_name == 'InterHT':
            self.u = 1

        self.entity_embedding = tf.Variable(tf.zeros([nentity, self.entity_dim]), trainable=True)
        self.relation_embedding = tf.Variable(tf.zeros([nrelation, self.relation_dim]), trainable=True)

        initializer_range = (self.gamma.numpy() + self.epsilon) / hidden_dim
        initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)

        self.entity_embedding.assign(initializer(self.entity_embedding.shape))

        initializer = tf.random_uniform_initializer(-initializer_range, initializer_range)
        self.relation_embedding.assign(initializer(self.relation_embedding.shape))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    def call(self, sample, training=True, **kwargs):
        sample, mode = sample

        def single_mode():
            positive_sample, negative_sample = sample
            batch_size, negative_sample_size = tf.shape(positive_sample)[0], 1

            head = tf.gather(self.entity_embedding, positive_sample[:, 0])
            head = tf.expand_dims(head, axis=1)

            relation = tf.gather(self.relation_embedding, positive_sample[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, positive_sample[:, 2])
            tail = tf.expand_dims(tail, axis=1)

            return head, relation, tail

        def head_batch_mode():
            tail_part, head_part = sample
            batch_size, negative_sample_size = tf.shape(head_part)[0], tf.shape(head_part)[1]

            head = tf.gather(self.entity_embedding, tf.reshape(head_part, [-1]))
            head = tf.reshape(head, [batch_size, negative_sample_size, -1])

            relation = tf.gather(self.relation_embedding, tail_part[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, tail_part[:, 2])
            tail = tf.expand_dims(tail, axis=1)

            return head, relation, tail

        def tail_batch_mode():
            head_part, tail_part = sample
            batch_size, negative_sample_size = tf.shape(tail_part)[0], tf.shape(tail_part)[1]

            head = tf.gather(self.entity_embedding, head_part[:, 0])
            head = tf.expand_dims(head, axis=1)

            relation = tf.gather(self.relation_embedding, head_part[:, 1])
            relation = tf.expand_dims(relation, axis=1)

            tail = tf.gather(self.entity_embedding, tf.reshape(tail_part, [-1]))
            tail = tf.reshape(tail, [batch_size, negative_sample_size, -1])

            return head, relation, tail

        def default_mode():
            head = tf.gather(self.entity_embedding, sample[0][:, 0])
            head = tf.expand_dims(head, axis=1)
            return head, head, head

            raise ValueError('mode %s not supported' % mode)

        head, relation, tail = tf.cond(tf.equal(mode, b'single'), lambda: single_mode(),
                                       lambda: tf.cond(tf.equal(mode, b'head-batch'), lambda: head_batch_mode(),
                                                       lambda: tf.cond(tf.equal(mode, b'tail-batch'),
                                                                       lambda: tail_batch_mode(),
                                                                       lambda: default_mode())))

        model_func = {
            'InterHT': self.InterHT
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def InterHT(self, head, relation, tail, mode):
        a_head, b_head = tf.split(head, num_or_size_splits=2, axis=2)
        re_head, re_mid, re_tail = tf.split(relation, num_or_size_splits=3, axis=2)
        a_tail, b_tail = tf.split(tail, num_or_size_splits=2, axis=2)

        e_h = tf.ones_like(b_head)
        e_t = tf.ones_like(b_tail)

        a_head = tf.linalg.normalize(a_head, ord=2, axis=-1)[0]
        a_tail = tf.linalg.normalize(a_tail, ord=2, axis=-1)[0]
        b_head = tf.linalg.normalize(b_head, ord=2, axis=-1)[0]
        b_tail = tf.linalg.normalize(b_tail, ord=2, axis=-1)[0]
        b_head = b_head + self.u * e_h
        b_tail = b_tail + self.u * e_t

        score = a_head * b_tail - a_tail * b_head + re_mid
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score

    @tf.function
    def train_step_fn(self, inputs):
        positive_sample, negative_sample, subsampling_weight, mode = inputs
        with tf.GradientTape() as tape:
            negative_score = self(((positive_sample, negative_sample), mode[0]), training=True)
            negative_score = tf.reduce_sum(
                tf.nn.softmax(negative_score * 1, axis=1) * tf.math.log_sigmoid(-negative_score), axis=1)
            positive_score = self(((positive_sample, negative_sample), b'single'))
            positive_score = tf.squeeze(tf.math.log_sigmoid(positive_score), axis=1)
            positive_sample_loss = -tf.reduce_sum(subsampling_weight * positive_score) / tf.reduce_sum(
                subsampling_weight)
            negative_sample_loss = -tf.reduce_sum(subsampling_weight * negative_score) / tf.reduce_sum(
                subsampling_weight)
            loss = (positive_sample_loss + negative_sample_loss) / 2

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"custom_loss": loss}