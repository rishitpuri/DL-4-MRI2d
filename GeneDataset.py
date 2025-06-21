import torch
import pandas as pd
from torch.utils.data import Dataset

class GeneDataset(Dataset):
    """A torch dataset for loading gene data using a dataframe."""

    def __init__(self, root):
        """Initialize the dataset with gene data and labels.

        Args:
            root (string): Path to the CSV file containing gene data and labels.
        """
        self.root = root
        self.data = self.load_data()

    def load_data(self):
        """Load the pandas dataframe from the provided CSV path."""
        dataframe = pd.read_csv(self.root)
        return dataframe

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Retrieve gene data and its label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the gene data tensor and label.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # Extract gene data starting from the fourth column (index 3)
        genes = self.data.iloc[index, 3:].to_numpy(dtype=float)
        genes = torch.tensor(genes, dtype=torch.float32)
        label = self.data.iloc[index, 2]  # Assuming 'Label' is the third column

        sample = {'genes': genes, 'label': label}
        return sample
