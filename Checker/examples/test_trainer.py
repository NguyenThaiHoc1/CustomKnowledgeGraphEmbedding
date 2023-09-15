import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import Mean
import tensorflow as tf
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('/content/CustomKnowledgeGraphEmbedding')
sys.path.append('/content/CustomKnowledgeGraphEmbedding/KnowledgeGraphEmbedding')
# sys.path.append(parent)
from trainer_checker import test_trainer
from tensorflow_codes.supervisor import getTFTrainer
from codes.model import getTorchTrainer

# input_size = 10
# hidden_size = 20
# output_size = 5

# class SimpleModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleModel, self).__init__()
#         self.dense_1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.dense_2 = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)
#     def forward(self, x):
#         x = self.dense_1(x)
#         x = self.relu(x)
#         x = self.dense_2(x)
#         # x = self.softmax(x)
#         return x

# torch_model = SimpleModel(input_size, hidden_size, output_size)

# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,), name='dense_1'),
#     tf.keras.layers.Dense(output_size, activation='softmax', name='dense_2')
# ])

# class TorchTrainer:
#     def __init__(self, model):
#         self.model = model
#         self.loss = nn.CrossEntropyLoss()
#         self.loss_result = Mean()
#         self.optimizer = optim.Adam(model.parameters(), lr=0.001)
#         self.metrics = {'acc': MulticlassAccuracy()}
        
#     def train_step(self, inputs):
#         inputs, labels = inputs
#         self.optimizer.zero_grad()
#         outputs = self.model(inputs)
#         loss = self.loss(outputs, labels)
#         loss.backward()
#         self.optimizer.step()
#         #update metrics
#         for metric_name, metric in self.metrics.items():
#           metric.update(outputs, labels)
#         self.loss_result.update(loss)
#         # result
#         result = { metric_name: metric.compute() for metric_name, metric in self.metrics.items()}
#         result['loss'] = self.loss_result.compute()
#         return result
#     def train(self, train_loader, num_epochs):
#         self.model.train()
#         for epoch in range(num_epochs):
#             for inputs in train_loader:
#                 result = self.train_step(inputs)
#             print(f"Epoch {epoch+1}/{num_epochs}, result: {result}")

#     def test_step(self, inputs):
#         inputs, labels = inputs
#         with torch.no_grad():
#             outputs = self.model(inputs)
#             #update metrics
#             loss = self.loss(outputs, labels)
#             for metric_name, metric in self.metrics.items():
#                 metric.update(outputs, labels)
#             self.loss_result.update(loss)
#         # result
#         result = { metric_name: metric.compute() for metric_name, metric in self.metrics.items()}
#         result['loss'] = self.loss_result.compute()
#         return result
#     def evaluate(self, test_loader):
#         self.model.eval()
#         for inputs in test_loader:
#             result = self.test_step(inputs)
#         print(f"Test evaluate: {result}")

# torch_trainer = TorchTrainer(torch_model)
# tf_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
#     )

# # Dataloader
# num_samples = 1000
# input_size = 10
# num_class = 5
# batch_size = 8

# X = np.random.rand(num_samples, input_size).astype(np.float32)
# y = np.random.randint(0, num_class, size=num_samples)

# # Torch dataloader
# X_train_tensor = torch.tensor(X)
# y_train_tensor = torch.tensor(y)
# torch_dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
# torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

tf_trainer, tf_model, tf_optimizer, tf_dataloader =getTFTrainer() 
torch_trainer, torch_model, torch_optimizer = getTorchTrainer() 
# Test 
test_trainer(
  tf_trainer, torch_trainer, 
  tf_model, torch_model, 
  tf_optimizer, torch_optimizer,
  tf_dataloader=tf_dataloader)
# test_trainer_from_pickle(
#     tf_trainer_path, torch_trainer_path, 
#     tf_dataloader_path=tf_dataloader_path, torch_dataloader_path=torch_dataloader_path)
