import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from model import DenseNetFeaturesBare
from transformer import EncoderLayer
from transformer import MultiHeadAttention

class SIGNet(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SIGNet, self).__init__()
        self.image_net = DenseNetFeaturesBare()
        self.enc_net = EncoderLayer(d_model,num_heads,2048,0.1)
        self.initial_emb = nn.Linear(1,16)
        # self.attn = MultiHeadAttention(d_model,num_heads)
        self.downsize = nn.Linear(8000, 512)
        self.fc1 = nn.Linear(512,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,3)
        self.relu = nn.ReLU()

    def pairWiseMul(self, x, y):
        # Perform element-wise multiplication
        return x * y

    def forward(self, image, gene):
        x = self.image_net(image)  # returns B x 512 dims flattened
        
        gene = gene.unsqueeze(-1) # Adds an additional dimension. B x 500 x 1
        # print(gene.shape)
        y = self.initial_emb(gene) # Returns B x 500 x16
        # print(y.shape) 
        y = self.enc_net(y, None)  # returns B x 500 x 16
        
        # Flatten the output of enc_net
        y = y.view(y.size(0), -1)  # Flatten to B x (500 * 16) = B x 8000
        
        # Downsize y to match the dimension of x (B x 512)
        y = self.downsize(y)  # Downsize to B x 512
        
        # Element-wise multiplication
        z = self.pairWiseMul(x, y)  # Element-wise multiply (B x 512)
        
        # Feedforward neural network layers
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.fc3(z)  # Final output layer (B x 3)
        
        return z