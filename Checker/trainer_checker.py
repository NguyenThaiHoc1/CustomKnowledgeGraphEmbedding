from tqdm import tqdm
import torch
from copy_weights import W_TF2Torch, W_Torch2TF, get_tf_weights, get_torch_weights
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset, DataLoader

def Torch_loader2TF_loader(torch_dataloader, batch_size, length=10e30):
  data = []
  for i, x in enumerate(torch_dataloader):
    data.append([tensor.numpy() for tensor in x])
    if i >= length*batch_size:
      break
  data = [list(x) for x in zip(*data)]
  return tf.data.Dataset.from_tensor_slices(tuple(data)).batch(batch_size)

def TF_loader2Torch_loader(tf_dataloader, batch_size, length=10e30):
  unbatch_tf_dataloader = tf_dataloader.unbatch()
  data = []
  for i, x in enumerate(unbatch_tf_dataloader):
    data.append(tuple(tensor.numpy() for tensor in x))
    if i >= length*batch_size:
      break
  return DataLoader(MyTorchDataset(data), batch_size=batch_size)


def SetZeroLearningRate(torch_optimizer, tf_optimizer):
  new_lr = 0.
  for param_group in torch_optimizer.param_groups:
    param_group['lr'] = new_lr
  # tf_optimizer.learning_rate.assign(new_lr)
  tf.keras.backend.set_value(tf_optimizer.learning_rate, new_lr)

def test_trainer(
    tf_trainer, torch_trainer, 
    tf_model, torch_model,
    tf_optimizer, torch_optimizer,
    tf_train_loader = None, torch_train_loader = None,
    tf_test_loader = None, torch_test_loader = None,
    train_metrics = ['loss'], test_metrics = ['full'],
    batch_size = 16, loader_length=10,
    ):
  is_train = tf_train_loader or torch_train_loader
  is_test = tf_test_loader or torch_test_loader
  W_Torch2TF(torch_model, tf_model)
  print("Copy weights TF2Torch passed !!\n")
  SetZeroLearningRate(torch_optimizer, tf_optimizer)
  print("SetZeroLearningRate passed !!\n")
  # Check dataloader
  if torch_train_loader: 
    tf_train_loader = Torch_loader2TF_loader(torch_train_loader, batch_size, loader_length)
  if tf_train_loader: 
    torch_train_loader = TF_loader2Torch_loader(tf_train_loader, batch_size, loader_length)
    tf_train_loader = tf_train_loader.unbatch().batch(batch_size)
  if torch_test_loader: 
    tf_test_loader = Torch_loader2TF_loader(torch_test_loader, batch_size, loader_length)
  if tf_test_loader: 
    torch_test_loader = TF_loader2Torch_loader(tf_test_loader, batch_size, loader_length)
    tf_test_loader = tf_test_loader.unbatch().batch(batch_size)
  print("Dataloader passed !!\n")
  # tf_train_loader = tf_train_loader.unbatch().batch(batch_size)
  trainer_checker = TrainerChecker(
    torch_trainer, tf_trainer, 
    train_metrics, test_metrics,
    loader_length, batch_size
    )
  if is_train:
    if not trainer_checker.check_train_step(tf_train_loader, torch_train_loader):
      return False
  if is_test:
    if not trainer_checker.chek_test_step(tf_test_loader, torch_test_loader):
      return False
  return True

class MyTorchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class TrainerChecker:
    def __init__(self, torch_trainer, tf_trainer,
                  train_metrics, test_metrics,
                  length, batch_size):
        self.torch_trainer = torch_trainer
        self.tf_trainer = tf_trainer
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.length = length
        self.batch_size = batch_size       
        pass
    def check_train_step(self, tf_dataloader, torch_dataloader):
        torch_iter = iter(torch_dataloader)
        tf_iter = iter(tf_dataloader)
        for i in tqdm(range(self.length), total=self.length):
          torch_data, tf_data = next(torch_iter), next(tf_iter)
          tf_metrics = self.tf_trainer.train_step(tf_data)#['loss'].numpy()
          torch_metrics = self.torch_trainer.train_step(torch_data)#['loss'].detach().numpy()
          for metric_name in self.train_metrics:
            if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name]):
              print("Error: Different train_step !!\n")
              return False
        print("Check train_step passed !!\n")
        return True

    def chek_test_step(self, tf_dataloader, torch_dataloader):  
        tf_metrics_names = [ metric.name for metric in self.tf_trainer.metrics]
        tf_metrics = self.tf_trainer.evaluate(tf_dataloader, steps=self.length)
        tf_metrics = dict(zip(tf_metrics_names , tf_metrics))
        torch_metrics = self.torch_trainer.evaluate(torch_dataloader, steps=self.length)
        if self.test_metrics[0] == 'full': # type: ignore
          self.test_metrics = set(torch_metrics.keys()) & set(tf_metrics.keys())
        for metric_name in self.test_metrics:
            if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name], rtol=1e-3, atol=1e-3):
              print("Error: Different test_step !!\n")
              return False
        print("\nCheck test_step passed !!\n")
        return True