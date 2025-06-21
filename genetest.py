import allel
import numpy as np
import pandas as pd

# Read VCF file using allel
vcf_path = '/N/project/SingleCell_Image/Nischal/Dilip/FINAL-FILES/VCF_useThis.vcf'
callset = allel.read_vcf(vcf_path)

# Extract genotype array (shape: variants x samples x ploidy)
genotypes = callset['calldata/GT']

# Convert genotypes to a 2D matrix (collapsing ploidy into a single number)
# For diploid organisms: 0/0 -> 0, 0/1 -> 1, 1/1 -> 2
genotype_numeric = allel.GenotypeArray(genotypes).to_n_alt()

# Convert this to a DataFrame (for easier manipulation)
genotype_df = pd.DataFrame(genotype_numeric)
# print(genotype_df.head())
print(genotype_df.T.shape)