import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('/content/CustomKnowledgeGraphEmbedding/')
sys.path.append('/content/CustomKnowledgeGraphEmbedding/KnowledgeGraphEmbedding')
from trainer_checker import test_trainer
from tensorflow_codes.supervisor import getTFTrainer
from codes.model import getTorchTrainer

tf_trainer, tf_model, tf_optimizer, tf_dataloader, tf_test_loader =getTFTrainer() 
torch_trainer, torch_model, torch_optimizer = getTorchTrainer() 
# Test 
test_trainer(
  tf_trainer, torch_trainer, 
  tf_model, torch_model, 
  tf_optimizer, torch_optimizer,
  # tf_train_loader=tf_dataloader,
  tf_test_loader=tf_test_loader,
  batch_size=1, 

  )
