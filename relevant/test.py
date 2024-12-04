import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_paths = [
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNDBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
]

# Load the datasets
dataframes = {name: pd.read_excel(path) for path, name in file_paths}

# Compute correlation matrices
correlation_matrices = {name: df.select_dtypes(include=[float, int]).corr() for name, df in dataframes.items()}

# Plot correlation matrices
for name, corr_matrix in correlation_matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix for {name}')
    plt.show()

# Plot pairplot for each dataset
for name, df in dataframes.items():
    numeric_df = df.select_dtypes(include=[float, int])
    sns.pairplot(numeric_df)
    plt.suptitle(f'Pairplot for {name}', y=1.02)
    plt.show()