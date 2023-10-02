import tensorflow as tf
import os
from utils import parse_tfrecord_fn, reshape_function


def read_info(path_file):
    with open(path_file, "r") as f:
        nrelation, nentity = f.read().rstrip('\n').split(' ')
    return int(nrelation), int(nentity)


def read_dataloader(dataloader, show_shape=True):
    count_split = 0
    for features in dataloader:
        data = features
        a, b, c, d = data
        if show_shape:
            print(a.shape, b.shape, c.shape, d.shape)
            break
        count_split += 1
    print(count_split)


if __name__ == '__main__':
    folder_path = 'split_data/wn18rr'
    folder_mode = 'valid'
    folder_data_path = os.path.join(folder_path, folder_mode)

    # list all file tfrec
    file_list = os.listdir(folder_data_path)
    tfrec_files = [os.path.join(folder_data_path, file) for file in file_list if file.endswith("head.tfrec")]
    print("List file tfrec: \n", tfrec_files)
    raw_dataset = tf.data.TFRecordDataset(tfrec_files)
    dataloader = raw_dataset.map(parse_tfrecord_fn)
    dataloader = dataloader.map(lambda inputs: reshape_function(inputs, batch_size=512))

    nrelation, nentity = read_info(path_file=os.path.join(folder_path, "info.txt"))
    print(nrelation, " ", nentity)

    read_dataloader(dataloader, show_shape=False)
