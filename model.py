import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Returns a 512 dimensional image embedding (Flattened)

class DenseNetFeatures(nn.Module):
    def __init__(self) -> None:
        super(DenseNetFeatures, self).__init__()
        self.dense = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)

        modules = list(self.dense.features.children())[1:]

        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0.weight.data = self.dense.features.conv0.weight.data.mean(dim=1, keepdim=True)
        
        self.features = nn.Sequential(
            self.conv0,
            *modules
        )
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to 1x1

        # Fully connected layer
        self.fc1 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class DenseNetFeaturesOnly(nn.Module):
    def __init__(self) -> None:
        super(DenseNetFeaturesOnly, self).__init__()
        self.dense = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)

        modules = list(self.dense.features.children())[1:]

        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0.weight.data = self.dense.features.conv0.weight.data.mean(dim=1, keepdim=True)
    
        self.features = nn.Sequential(
            self.conv0,
            *modules
        )
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to 1x1

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNetFeatures(nn.Module):
    def __init__(self) -> None:
        super(ResNetFeatures, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the first and the last layer from ResNet 50
        modules = list(self.resnet50.children())[1:-1]
        
        # Copy the weights from the original conv1 layer (which has 3 input channels)
        # by averaging the weights across the 3 RGB channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = self.resnet50.conv1.weight.data.mean(dim=1, keepdim=True)

        self.resnet50 = nn.Sequential(
            self.conv1,
            *modules)
        
        self.fc1 = nn.Linear(2048,512)
        
    def forward(self,x):
        x = self.resnet50(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x

class TestNet(nn.Module):
    def __init__(self) -> None:
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(128*24*24, 2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(64,3)
    
    def forward(self, x):
        x = self.maxpool(self.norm1(F.relu(self.conv1(x))))
        x = self.maxpool(self.norm2(F.relu(self.conv2(x))))
        x = self.maxpool(self.norm3(F.relu(self.conv3(x))))

        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        # x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x