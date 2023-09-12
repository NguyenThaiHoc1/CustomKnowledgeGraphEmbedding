import os


class RegularDataRaw:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    @staticmethod
    def _read_triple(file_path, entity2id, relation2id):
        triples = []
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    def _read_specificate(self, data_path):
        with open(os.path.join(data_path, 'entities.dict')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(data_path, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triples = self._read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
        valid_triples = self._read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
        test_triples = self._read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
        return train_triples, valid_triples, test_triples, entity2id, relation2id

    def read(self):
        return self._read_specificate(data_path=self.input_dir)
