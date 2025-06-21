import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from model import DenseNetFeaturesOnly
from transformer import EncoderLayer
from transformer import MultiHeadAttention

class ImagingNet(nn.Module):
    def __init__(self):
        super(ImagingNet, self).__init__()
        self.image_net = DenseNetFeaturesOnly()
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image):
        x = self.image_net(image)  # returns B x 1024 dims flattened
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class GeneTransformer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(GeneTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, 2048, 0.1) for _ in range(2)])
        self.initial_emb = nn.Linear(64,16)
        self.downsize = nn.Linear(157008, 4096)
        self.fc1 = nn.Linear(4096,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,256)
        self.fc4 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.3)

    def forward(self, gene):

        print(gene.shape)
        # gene = gene.unsqueeze(-1) # Adds an additional dimension. B x 9813 x 64
        
        y = self.initial_emb(gene) # Returns B x 9813 x 16
        
        # Returns B x 9813 x 16
        for enc in self.encoder_layers:
            y = enc(y, None)
        
        y = y.view(y.size(0), -1)  # Flatten to B x (9813 * 16) = B x 157008
       
        # Downsize y to match the dimension of x (B x 4096)
        y = self.downsize(y)

        # FeedForward with dropout
        z = self.Dropout(self.relu(self.fc1(y)))
        z = self.Dropout(self.relu(self.fc2(z)))
        z = self.Dropout(self.relu(self.fc3(z)))
        z = self.fc4(z)  # Final output layer (B x 3)

        return z