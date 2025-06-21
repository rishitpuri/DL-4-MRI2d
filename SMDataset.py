import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

"""
Structure of pandas dataframe for reference.
"IID" "Image Path" "Label" "PC1" "PC2" ..... "PC500" 
"""

class SMDataset(Dataset):
    """A torch dataset for loading gene data (*S*NP) and image (*M*RI) using a dataframe."""

    def __init__(self, root, transform=None):
        """Initialize the pandas dataframe containing the data.

        Args:
            root (string): Path to the csv containing image paths and gene data.
            transform (callable, optional): Apply transforms on image if required. Defaults to None.
        """
        self.root = root
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        """Load the pandas dataframe from the provided path."""
        dataframe = pd.read_csv(self.root)
        return dataframe

    def __len__(self):
        """Display the size of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Fetches item one at a time or in batches.

        Args:
        index (int, tensor): The index of the required entry from the dataset.

        Returns:
        dict: Contains a dict containing the image(PIL image), genetic data(torch tensor) and label(int).
        """
        if torch.is_tensor(index):
            index = index.tolist()

        genes = self.data.iloc[index,3:].to_numpy(dtype=float)
        genes = torch.tensor(genes, dtype=torch.float32)
        label = self.data.iloc[index,2]

        try:
            img_path = self.data.iloc[index,1]
            image = Image.open(img_path).convert('L')

        except Exception as e:
            print(f"Unable to load image at {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image)

        sample = {'image':image, 'genes':genes, 'label':label}

        return sample

"""
Example Usage:

dataset = SMDataset('path_to_csv')

print(dataset[0])
"""