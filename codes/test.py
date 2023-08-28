import os
from collections import defaultdict
from tqdm import tqdm
from dataloader import TrainDataset
from torch.utils.data import DataLoader

data_path = "D:\hoc-nt\FJS\KGE\CustomKnowledgeGraphEmbedding\data\FB15k"
negative_sample_size = 1


def read_triple(file_path, entity2id, relation2id):
    '''
        Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


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

nentity = len(entity2id)
nrelation = len(relation2id)

train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)

# train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
# for i in tqdm(range(len(train_triples['head']))):
#     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
#     train_count[(head, relation)] += 1
#     # if not args.inverses:
#     #     train_count[(tail, -relation - 1)] += 1
#     train_true_head[(relation, tail)].append(head)
#     train_true_tail[(head, relation)].append(tail)

dataset = TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'head-batch')
# dataset = TrainDatasetHT(train_triples, nentity, nrelation, negative_sample_size, 'head-batch',
#                          train_count,
#                          train_true_head, train_true_tail),

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'head-batch'),
    batch_size=1,
    shuffle=True,
    collate_fn=TrainDataset.collate_fn
)

for data in train_dataloader_head:
    print(data)
