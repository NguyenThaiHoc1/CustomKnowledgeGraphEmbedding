import torch
from tqdm import tqdm
import torch
from copy_weights import W_TF2Torch
import tensorflow as tf
import numpy as np
import pickle

def SetZeroLearningRate(torch_optimizer, tf_optimizer):
  new_lr = 0.
  for param_group in torch_optimizer.param_groups:
    param_group['lr'] = new_lr
  tf_optimizer.learning_rate.assign(new_lr)

def test_trainer_from_pickle(
    tf_trainer_path, torch_trainer_path, 
    tf_dataloader_path=None, torch_dataloader_path=None):
  with open(tf_trainer_path, "rb") as f:
    tf_trainer = pickle.load(f)
  with open(torch_trainer_path, "rb") as f:
    torch_trainer = pickle.load(f)
  if tf_dataloader_path!=None:
    with open(tf_dataloader_path, "rb") as f:
      tf_dataloader = pickle.load(f)
  if torch_trainer_path!=None:
    with open(torch_trainer_path, "rb") as f:
      torch_dataloader = pickle.load(f)
  return test_trainer(
    tf_trainer, torch_trainer, 
    tf_dataloader=tf_dataloader,
    torch_dataloader=torch_dataloader,)

def test_trainer(
    tf_trainer, torch_trainer, 
    tf_model=None, torch_model=None,
    tf_dataloader = None,
    torch_dataloader = None,
    ):
  if tf_model==None:
    tf_model=tf_trainer.model
  if torch_model==None:
    torch_model=torch_trainer.model

  W_TF2Torch(tf_model, torch_model)
  print("Copy weights TF2Torch passed !!\n")
  SetZeroLearningRate(torch_trainer.optimizer, tf_trainer.optimizer)
  print("Check SetZeroLearningRate passed !!\n")

  trainer_checker = TrainerChecker(torch_trainer, tf_model, torch_dataloader=torch_dataloader, tf_dataloader=tf_dataloader)
  is_oke_train = trainer_checker.check_train_step()
  is_oke_test = trainer_checker.chek_test_step()
  return is_oke_train and is_oke_test

class TrainerChecker:
    def __init__(self, torch_trainer, tf_trainer,
                 torch_dataloader=None, tf_dataloader=None,
                 length=100):
        self.torch_trainer = torch_trainer
        self.tf_trainer = tf_trainer
        self.torch_dataloader = torch_dataloader
        self.tf_dataloader = tf_dataloader
        self.length = length
        self.torch_iter = None
        self.tf_iter = None

    def resetIter(self):
        if self.torch_dataloader:
            self.torch_iter = iter(self.torch_dataloader)
        if self.tf_dataloader:
            self.tf_iter = iter(self.tf_dataloader)

    def getNextData(self):
        if self.torch_dataloader:
            torch_data = next(self.torch_iter)
        if self.tf_dataloader:
            tf_data = next(self.tf_iter)
        if not self.torch_dataloader:
            torch_data = tuple(torch.from_numpy(tensor.numpy()) for tensor in tf_data)
        if not self.tf_dataloader:
            tf_data = tuple(tf.convert_to_tensor(tensor.numpy()) for tensor in torch_data)
        return torch_data, tf_data

    def check_train_step(self):
        self.resetIter()
        for _ in tqdm(range(self.length)):
            torch_data, tf_data = self.getNextData()
            tf_loss = self.tf_trainer.train_step(tf_data)['loss'].numpy()
            torch_loss = self.torch_trainer.train_step(torch_data)['loss'].detach().numpy()
            if not np.allclose(torch_loss, tf_loss, rtol=1e-5, atol=1e-5):
                print("Error: Different train_step !!\n")
                return False
        print("Check train_step passed !!\n")
        return True

    def chek_test_step(self):
        self.resetIter()
        for _ in tqdm(range(self.length)):
            torch_data, tf_data = self.getNextData()
            torch_metrics = self.torch_trainer.test_step(torch_data)
            tf_metrics = self.tf_trainer.test_step(tf_data)
            common_metrics = set(torch_metrics.keys()) & set(tf_metrics.keys())
            for metric_name in common_metrics:
                if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name]):
                  print("Error: Different test_step !!\n")
                  return False
        print("\nCheck test_step passed !!\n")
        return True