import numpy as np
import cv2

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import DatasetFolder
from torch import Tensor

import os
import pandas as pd
import json
import numpy as np

from typing import Callable, List, Optional, Tuple, Union

class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=4096, out_features=2048)
        self.layer_2 = nn.Linear(in_features=2048, out_features=1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.relu(x)
        return x

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        n_way = len(torch.unique(support_labels))

        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        dists = torch.cdist(z_query, z_proto)
        scores = -dists

        return scores

#@title Proto Utils

maps = {0: 'Downy Mildew', # TEST
        1: 'blast', # TEST
        2: 'blight',
        3: 'leaf spot',
        4: 'mosaic virus',
        5: 'powdery mildew', # TEST
        6: 'rot',
        7: 'rust',
        8: 'smut',
        9: 'wilt'}

def euclidean_distance2(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def get_prototypes(prototypes_path):
  with open(prototypes_path, 'r') as json_file:
    serializable_embeddings = json.load(json_file)

  for k,embedding in serializable_embeddings.items():
    embedding = np.array(embedding)

  return serializable_embeddings

def init_proto(prototypes_path, ckpt_path):
  protos = get_prototypes(prototypes_path)
  convolutional_network =  ModelV0()
  model_proto = PrototypicalNetworks(convolutional_network).to("cpu")
  model_proto.load_state_dict(torch.load(ckpt_path))
  return protos, model_proto

def predict_class(sample, prototypes, model_proto):
  min_distance = float('inf')
  predicted_class = None
  vec = model_proto.backbone.forward(Tensor(sample))
  vec = vec.detach().numpy()

  for cls, prototype in prototypes.items():
      print(np.array(prototype).shape)

      distance = euclidean_distance2(vec, prototype)
      if distance < min_distance:
          min_distance = distance
          predicted_class = cls

  return maps[int(predicted_class)]