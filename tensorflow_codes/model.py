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

        self.model_func = {
            'InterHT': self.InterHT,
        }
        if model_name == 'TransD':
            self.dim_e = dim_e
            self.dim_r = dim_r
            self.pnorm = norm # default 2

            self.entity_embeddings = tf.keras.layers.Embedding(ent_total, dim_e)
            self.entity_projection_embeddings = tf.keras.layers.Embedding(ent_total, dim_e)
            self.relation_embeddings = tf.keras.layers.Embedding(rel_total, dim_r)
            self.relation_projection_embeddings = tf.keras.layers.Embedding(rel_total, dim_r)
        if model_name == 'RotateCT':
             #'soft_margin'
            self.r_phase = self.create_embedding(self.dim, emb_type="relation", name="r_phase", init_method="uniform", init_params=[-math.pi, math.pi])
        pass

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

            negative_tail_score = self.model_func[self.model_name](head, relation, tail, mode)
            return negative_tail_score

        head_score = head_batch_mode(sample, mode)
        tail_score = tail_batch_mode(sample, mode)
        negative_condition = tf.cond(tf.equal(mode, 0), lambda: 1.0, lambda: 0.0)
        return head_score * negative_condition + tail_score * (1 - negative_condition)

    def InterHT(self, head, relation, tail, mode):
        a_head, b_head = tf.split(head, num_or_size_splits=2, axis=2)
        _, re_mid, _ = tf.split(relation, num_or_size_splits=3, axis=2)
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

    def DistMult(self, head, relation, tail, mode):
        if mode == 0:
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        score = tf.reduce_sum(score, axis=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = tf.split(head, num_or_size_splits=2, axis=2)
        re_relation, im_relation = tf.split(relation, num_or_size_splits=2, axis=2)
        re_tail, im_tail = tf.split(tail, num_or_size_splits=2, axis=2)
        if mode == 0:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
        score = tf.reduce_sum(score, axis=2)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 0:
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score

    def TransD(self, head, relation, tail, mode):
        def _transfer(e, ep, rp):
            i = tf.eye(rp.shape[-1], e.shape[-1])
            rp = tf.expand_dims(rp, axis=-1)
            ep = tf.expand_dims(ep, axis=1)
            m = tf.matmul(rp, ep)
            result = tf.matmul(m + i,  tf.expand_dims(e, axis=-1))
            return tf.squeeze(result, axis=-1)
        
        head, p_head = tf.split(head, num_or_size_splits=2, axis=2)
        relation, p_relation = tf.split(relation, num_or_size_splits=2, axis=2)
        tail, p_tail = tf.split(tail, num_or_size_splits=2, axis=2)
    
        head_transfer = _transfer(head, p_head, p_relation)
        tail_transfer = _transfer(tail, p_tail, p_relation)
        return tf.norm(head_transfer + relation - tail_transfer, axis=-1, ord=self.pnorm) ** 2



    # def STransE(self, head, relation, tail, mode):
    #     """f_r(h,t) = |Mr1*h+r-Mr2*t| constraints on the norm <=1"""
    #     # Projection matrix Mr, shape (k, k), initialize with identity matrix.
    #     self.Mr1 = tf.get_variable("Mr1", [self.k, self.k], initializer=tf.initializers.identity(gain=0.1))
    #     self.Mr1 = tf.tile(tf.expand_dims(self.Mr1, 0), [self.b, 1, 1])  # (b, k, k)
    #     self.Mr2 = tf.get_variable("Mr2", [self.k, self.k], initializer=tf.initializers.identity(gain=0.1))
    #     self.Mr2 = tf.tile(tf.expand_dims(self.Mr2, 0), [self.b, 1, 1])  # (b, k, k)

    #     h = tf.expand_dims(h, axis=2)  # (b, k) -> (b, k, 1)
    #     t = tf.expand_dims(t, axis=2)  # (b, k) -> (b, k, 1)
    #     dis = tf.squeeze(tf.matmul(self.Mr1, h), axis=2) + r + tf.squeeze(tf.matmul(self.Mr2, t), axis=2)
    #     if self.params.score_func.lower() == 'l1':  # L1 score
    #         score = tf.reduce_sum(tf.abs(dis), axis=1)
    #     elif self.params.score_func.lower() == 'l2':  # L2 score
    #         score = tf.sqrt(tf.reduce_sum(tf.square(dis), axis=1))
    #     return score


    def TripleRE(self, head, relation, tail, mode):
        re_head, re_mid, re_tail = tf.split(relation, num_or_size_splits=3, axis=2)

        head = tf.math.l2_normalize(head, axis=-1)
        tail = tf.math.l2_normalize(tail, axis=-1)

        re_head = tf.math.l2_normalize(re_head, axis=-1)
        re_tail = tf.math.l2_normalize(re_tail, axis=-1)
        re_head = re_head * tf.sqrt(re_head.shape[-1])
        re_tail = re_tail * tf.sqrt(re_tail.shape[-1])

        score = head * re_head - tail * re_tail + re_mid
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score

    def TranS(self, head, relation, tail, mode):
        a_head, b_head = tf.split(head, num_or_size_splits=2, axis=2)
        re_head, re_mid, re_tail = tf.split(relation, num_or_size_splits=3, axis=2)
        a_tail, b_tail = tf.split(tail, num_or_size_splits=2, axis=2)

        e_h = tf.ones_like(b_head) # can k ta ??
        e_t = tf.ones_like(b_tail)

        a_head = tf.linalg.normalize(a_head, ord=2, axis=-1)[0] # [0] ???
        a_tail = tf.linalg.normalize(a_tail, ord=2, axis=-1)[0]
        b_head = tf.linalg.normalize(b_head, ord=2, axis=-1)[0]
        b_tail = tf.linalg.normalize(b_tail, ord=2, axis=-1)[0]
        re_head = tf.linalg.normalize(re_head, ord=2, axis=-1)[0]
        re_tail = tf.linalg.normalize(re_tail, ord=2, axis=-1)[0]

        b_head = b_head + self.u * e_h
        b_tail = b_tail + self.u * e_t

        score = a_head * b_tail - a_tail * b_head + a_head * re_head + a_tail * re_tail + re_mid
        score = self.gamma - tf.norm(score, ord=1, axis=2)
        return score

    
    def RotPro(self, head, relation, tail, mode):
        re_head, im_head = tf.split(head, num_or_size_splits=2, axis=2)
        r_phase, proj_a, proj_b, proj_phase = tf.split(relation, num_or_size_splits=4, axis=2)
        re_tail, im_tail = tf.split(tail, num_or_size_splits=2, axis=2)
        
        re_r_phase = tf.ones_like(r_phase) * tf.cos(r_phase) 
        im_r_phase = tf.ones_like(r_phase) * tf.sin(r_phase)
        # r_phase = tf.complex(re_r_phase, im_r_phase)

        re_proj_phase = tf.ones_like(proj_phase) * tf.cos(proj_phase)
        im_proj_phase = tf.ones_like(proj_phase) * tf.sin(proj_phase)
        # proj_phase = tf.complex(re_proj_phase, im_proj_phase)

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

        return tf.norm(hr - tt, ord=1, axis=-1)  # Default: 1 (like RotatE).


    def RotateCT(self, head, relation, tail, mode):
        re_head, im_head = tf.split(head, num_or_size_splits=2, axis=2)
        re_b, im_b = tf.split(relation, num_or_size_splits=2, axis=2)
        re_tail, im_tail = tf.split(tail, num_or_size_splits=2, axis=2)
        r_phase = self.r_phase

        c_head = tf.complex(re_head, im_head)
        c_tail = tf.complex(re_tail, im_tail)
        c_b = tf.complex(re_b, im_b)
        c_r = tf.complex(
            tf.ones_like(r_phase) * tf.cos(r_phase),
            tf.ones_like(r_phase) * tf.sin(r_phase))

        score = (c_head - c_b) * c_r - (c_tail - c_b)
        score = tf.norm(score, ord=1, axis=-1)  # Default: norm 1
        return score