import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """A torch dataset for loading image data using a dataframe."""

    def __init__(self, root, transform=None):
        """Initialize the dataset with image paths and labels.

        Args:
            root (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample. Defaults to None.
        """
        self.root = root
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        """Load the pandas dataframe from the provided CSV path."""
        dataframe = pd.read_csv(self.root)
        return dataframe

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Retrieve an image and its label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and label.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        label = self.data.iloc[index, 2]  # Assuming 'Label' is the third column
        img_path = self.data.iloc[index, 1]  # Assuming 'Image Path' is the second column

        try:
            image = Image.open(img_path).convert('L')  # Convert image to grayscale
        except Exception as e:
            print(f"Unable to load image at {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample
