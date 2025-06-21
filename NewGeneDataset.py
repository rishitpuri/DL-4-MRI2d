import torch
import pandas as pd
from torch.utils.data import Dataset
import allel
import numpy as np

class NewGeneDataset(Dataset):
    """A torch dataset for loading gene data using a dataframe."""

    def __init__(self, root):
        """Initialize the dataset with gene data and labels.

        Args:
            root (string): Path to the CSV file containing gene data and labels.
        """
        self.root = root
        self.load_data()

    def load_data(self):
        """Load the pandas dataframe from the provided CSV path."""
        
        vcf_path = '/N/project/SingleCell_Image/Nischal/Dilip/FINAL-FILES/VCF_useThis.vcf'
        callset = allel.read_vcf(vcf_path)
        genotypes = callset['calldata/GT']
        genotype_numeric = allel.GenotypeArray(genotypes).to_n_alt()
        genotype_df = pd.DataFrame(genotype_numeric)

        dataframe = pd.read_csv(self.root)

        self.snp_df = genotype_df.T
        self.phenotype_df = dataframe

    def __len__(self):
        """Return the total number of samples."""
        # print(self.phenotype_df.T.shape)
        return self.phenotype_df.T.shape[1]

    def __getitem__(self, index):
        """Retrieve SNP data and its phenotype label by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the SNP data tensor and phenotype label.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # Extract SNP data for the given index and convert into 9813 tuples of 64 SNPs each
        snp_row = self.snp_df.iloc[index].values
        snp_vector = [tuple(snp_row[i:i+64]) for i in range(0, len(snp_row), 64)]
        snp_vector_tensor = torch.tensor(snp_vector, dtype=torch.float32)

        # Retrieve the phenotype label for the given index
        phenotype = self.phenotype_df.iloc[index, 1]  # Assuming 'phenotype' is the second column
        phenotype_tensor = torch.tensor(phenotype, dtype=torch.float32)

        # Return as a dictionary to match the desired structure
        sample = {'genes': snp_vector_tensor, 'label': phenotype_tensor}
        return sample