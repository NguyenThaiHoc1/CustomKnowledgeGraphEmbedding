import os
from dataloader import DataLoader
from dataloader import DataGenerator
from dataloader import DataGenerator2Dataset, BidirectionalOneShotIterator, Bidirectional2Dataset
from model1 import TFKGEModel
import time
from time import sleep
from tqdm import tqdm
import psutil
import tensorflow as tf
from collections import namedtuple

EPOCHS = 10
data_path = "data/wn18rr"
# data_path = "D:\hoc-nt\FJS\KGE\CustomKnowledgeGraphEmbedding\data\wn18rr"

model = "InterHT"
hidden_dim = 1000
gamma = 24.0
double_relation_embedding = False
double_entity_embedding = True
triple_relation_embedding = True
learning_rate = 0.0001
TPU_WORKER = 1
regularization = 0.0
negative_adversarial_sampling = True
uni_weight = False
adversarial_temperature = 1.0

# Training ...
STEPS_PER_EPOCH = 10000
BATCH_SIZE = 8
negative_sample_size = 256


def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_data():
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

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    return train_triples, valid_triples, test_triples, entity2id, relation2id


def run_main():
    train_triples, valid_triples, test_triples, entity2id, relation2id = read_data()
    nentity = len(entity2id)
    nrelation = len(relation2id)

    # train
    train_generator_head = DataGenerator(
        train_triples, nentity, nrelation, negative_sample_size, 0,  # 'head-batch'
    )

    train_generator_tail = DataGenerator(
        train_triples, nentity, nrelation, negative_sample_size, 1,  # 'tail-batch'
    )

    train_dataset_head, train_length_head = DataGenerator2Dataset().convert(data_generator=train_generator_head)
    # train_dataset_tail, train_length_tail = DataGenerator2Dataset().convert(data_generator=train_generator_tail)

    # train_dataloader_head = DataLoader(train_dataset_head).gen_dataset(
    #     batch_size=16, is_training=True, shuffle=True,
    #     input_pipeline_context=None, preprocess=None,
    #     drop_remainder=False
    # )

    # train_dataloader_tail = DataLoader(train_dataset_tail).gen_dataset(
    #     batch_size=16, is_training=True, shuffle=True,
    #     input_pipeline_context=None, preprocess=None,
    #     drop_remainder=False
    # )

    # combined_dataset = tf.data.Dataset.sample_from_datasets(
    #     [train_dataloader_head, train_dataloader_tail],
    #     weights=[0.5, 0.5]
    # )

    combined_dataset = train_dataset_head
    combined_dataset = combined_dataset.repeat()  # the training dataset must repeat for several epochs
    combined_dataset = combined_dataset.shuffle(2048)
    combined_dataset = combined_dataset.batch(BATCH_SIZE, drop_remainder=True)  # slighly faster with fixed tensor sizes
    combined_dataset = combined_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return combined_dataset, nrelation, nentity


def compute_ram_cpu():
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
        while True:
            rambar.n = psutil.virtual_memory().percent
            cpubar.n = psutil.cpu_percent()
            rambar.refresh()
            cpubar.refresh()
            run_main()
            sleep(0.5)


def compute_ram_cpu_ver_rich():
    import time
    import psutil
    from rich.progress import Progress

    with Progress() as progress:
        rambar = progress.add_task("[red]ram%", total=100,
                                   completed=psutil.virtual_memory().percent,
                                   visible=False)
        cpubar = progress.add_task("[green]cpu%", total=100,
                                   completed=psutil.cpu_percent(),
                                   visible=False)

        while not progress.finished:
            progress.update(rambar, completed=psutil.virtual_memory().percent, advance=0.5)
            progress.update(cpubar, completed=psutil.cpu_percent(), advance=0.5)
            run_main()
            time.sleep(0.5)


@tf.function
def lrfn(epoch):
    if float(epoch) < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * float(epoch) + LR_START
    elif float(epoch) < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (float(epoch) - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:  # detect GPUs
    # strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

print("Number of accelerators: ", strategy.num_replicas_in_sync)

# Learning rate.
LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5.0
LR_SUSTAIN_EPOCHS = 0.0
LR_EXP_DECAY = .8
STEPS_PER_TPU_CALL = 99
VALIDATION_STEPS_PER_TPU_CALL = 29

dataloader, nrelation, nentity = run_main()
train_dist_ds = strategy.experimental_distribute_dataset(dataloader)

with strategy.scope():
    kge_model = TFKGEModel(
        model_name=model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=hidden_dim,
        gamma=gamma,
        double_entity_embedding=double_entity_embedding,
        double_relation_embedding=double_relation_embedding,
        triple_relation_embedding=triple_relation_embedding
    )


    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return lrfn(epoch=step // STEPS_PER_EPOCH)


    optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule())

    kge_model.compile(optimizer=optimizer,
                      loss=None,
                      metrics=None)

    kge_model.fit(
        train_dist_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS
    )
