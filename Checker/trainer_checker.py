import torch
from tqdm import tqdm
import torch
from copy_weights import W_TF2Torch, W_Torch2TF, get_tf_weights, get_torch_weights
import tensorflow as tf
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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
    tf_dataloader = None, torch_dataloader = None,
    train_metrics = ['loss'], test_metrics = ['full'],
    batch_size = 16, loader_length=100,
    ):
  assert torch_dataloader or tf_dataloader, "Both torch and tf dataloader none?"
  W_Torch2TF(torch_model, tf_model)
  print("Copy weights TF2Torch passed !!\n")
  SetZeroLearningRate(torch_optimizer, tf_optimizer)
  # torch_trainer.optimizer = optim.Adam(torch_model.parameters(), lr=0.)
  # tf_trainer.optimizer = tf.keras.optimizers.Adam(0.)
  print("Check SetZeroLearningRate passed !!\n")
  if not tf_dataloader: 
    tf_dataloader = Torch_loader2TF_loader(torch_dataloader, batch_size, loader_length)
  if not torch_dataloader: 
    torch_dataloader = TF_loader2Torch_loader(tf_dataloader, batch_size, loader_length)
  tf_dataloader = tf_dataloader.unbatch().batch(batch_size)
  # tf_dataloader = tf_dataloader.unbatch().batch(batch_size)
  trainer_checker = TrainerChecker(
    torch_trainer, tf_trainer, 
    torch_dataloader, tf_dataloader,
    train_metrics, test_metrics,
    loader_length, batch_size
    )
  is_oke_train = trainer_checker.check_train_step()
  # is_oke_test = trainer_checker.chek_test_step()
  # return is_oke_train and is_oke_test

class MyTorchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class TrainerChecker:
    def __init__(self, torch_trainer, tf_trainer,
                  torch_dataloader, tf_dataloader,
                  train_metrics, test_metrics,
                  length, batch_size):
        self.torch_trainer = torch_trainer
        self.tf_trainer = tf_trainer
        self.torch_dataloader = torch_dataloader
        self.tf_dataloader = tf_dataloader
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.length = length
        self.batch_size = batch_size       
        pass
    def check_train_step(self):
        self.torch_iter = iter(self.torch_dataloader)
        self.tf_iter = iter(self.tf_dataloader)
        for _ in tqdm(range(self.length), total=self.length):
            torch_data, tf_data = next(self.torch_iter), next(self.tf_iter)
            print('torch_data', torch_data)
            print('tf_data', tf_data)
            tf_loss = self.tf_trainer.train_step(tf_data)['loss'].numpy()
            torch_loss = self.torch_trainer.train_step(torch_data)['loss'].detach().numpy()
            if not np.allclose(torch_loss, tf_loss, rtol=1e-5, atol=1e-5):
                print("Error: Different train_step !!\n")
                print(torch_loss, tf_loss)
                return False
        print("Check train_step passed !!\n")
        return True

    def chek_test_step(self):
        self.torch_iter = iter(self.torch_dataloader)
        self.tf_iter = iter(self.tf_dataloader)        
        torch_metrics = self.torch_trainer.evaluate(self.torch_dataloader, steps=self.length)
        tf_metrics = self.tf_trainer.evaluate(self.tf_dataloader, steps=self.length)
        tf_metrics = dict(zip(self.tf_trainer.metrics_names , tf_metrics))
        if self.test_metrics[0] == 'full':
          self.test_metrics = set(torch_metrics.keys()) & set(tf_metrics.keys())
        for metric_name in self.test_metrics:
            if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name], rtol=1e-3, atol=1e-3):
              print("Error: Different test_step !!\n")
              return False
        print("\nCheck test_step passed !!\n")
        return True