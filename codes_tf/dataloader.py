from typing import Optional
import tensorflow as tf
import functools
import numpy as np

"""
    Luồng nó sẽ là 
    Loading data raw từ file lên 
    --> 
    Cho data_raw vào data sequence (__getitem__)
    --> 
    chuyển data sequence thành dataset 
    --> 
    chuyển dataset thành dataloader.
"""


def shard(dataset, input_pipeline_context):
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)
    return dataset


"""
    * DataLoader
    Nơi xử lý bên trong các tác vụ về batching or prefering ...
    hoặc có thể xử lý reprocess bên trong đây luôn nếu các tác vụ đơn giản 
    phức tạp thì cứ xử lý trước ở Data Sequence.
"""


class DataLoader(object):
    def __init__(self, dataset, size=None):
        self._dataset = dataset
        self._size = size

    @property
    def size(self) -> Optional[int]:
        return self._size

    def gen_dataset(self, batch_size=1, is_training=False, shuffle=False,
                    input_pipeline_context=None, preprocess=None,
                    drop_remainder=False):
        dataset = self._dataset
        dataset = shard(dataset, input_pipeline_context)

        if preprocess:
            preprocess = functools.partial(preprocess, is_training=is_training)
            dataset = dataset.map(preprocess, num_parallel_call=tf.data.AUTOTUNE)

        if is_training:
            if shuffle:
                buffer_size = 3 * batch_size
                if self._size:
                    buffer_size = min(self._size, buffer_size)
                dataset = dataset.shuffle(buffer_size=buffer_size)

        # dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        if self._size is not None:
            return self._size
        else:
            return len(self._dataset)


"""
    * DataGenerator
    Nơi xử lý dữ liệu đầu vào (tf Sequence)
"""


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = np.sqrt(1 / np.array([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = tf.convert_to_tensor(negative_sample)

        positive_sample = tf.convert_to_tensor(positive_sample)

        subsampling_weight = tf.convert_to_tensor(subsampling_weight)

        return positive_sample, negative_sample, subsampling_weight, self.mode


"""
    * DataGenerator2Dataset
    convert dữ liệu "đầu vào" thành dataset Nơi quy định format cũa tf.data.Dataset) ==> nó như là tf.from_tensor_slice
"""


class DataGenerator2Dataset:
    data_generator = None

    def _gen_data_generator(self):
        assert self.data_generator is not None, "Data Generator don't have anything."
        for i in range(self.data_generator.len):
            yield self.data_generator.__getitem__(i)

    def convert(self, data_generator):
        self.data_generator = data_generator

        dataset = tf.data.Dataset.from_generator(
            self._gen_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(3,), dtype=tf.int32),
                tf.TensorSpec(shape=(256,), dtype=tf.int32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        return dataset, len(self.data_generator)


"""
    cái này không có gì tùy bài toán.
"""


class Bidirectional2Dataset:
    data_generator = None

    def _gen_data_generator(self):
        assert self.data_generator is not None, "Data Generator don't have anything."
        for i in range(self.data_generator.len):
            yield self.data_generator.__getitem__(i)

    def convert(self, data_generator):
        self.data_generator = data_generator

        dataset = tf.data.Dataset.from_generator(
            self._gen_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 3,), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 128,), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 1,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.string)
            )
        )
        return dataset, self.data_generator.len


class BidirectionalOneShotIterator:
    def __init__(self, dataloader_head, dataloader_tail, train_length_head, train_length_tail):
        self.dataloader_head = self.one_shot_iterator(dataloader_head, train_length_head)
        self.dataloader_tail = self.one_shot_iterator(dataloader_tail, train_length_tail)
        self.train_length_head = train_length_head
        self.train_length_tail = train_length_tail
        self.step = 0
        self.len = self.train_length_tail + self.train_length_head

    def __len__(self):
        return self.train_length_tail + self.train_length_head

    def __getitem__(self, idx):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.dataloader_head)
        else:
            data = next(self.dataloader_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataset, size_dataset):
        dataset_iter = iter(dataset)
        for i in range(size_dataset):
            data = next(dataset_iter)
            yield data

