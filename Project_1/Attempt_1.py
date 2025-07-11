"""
Author: Carolyn Rentz
Assignment Title: Project 1: Getting a plot to show up
Date Created: 7/7/25
Date Last Modified: 7/9/25
"""
import traceback
import scanpy as sc
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import os as os


#List of .csv files
csv_files = ["6K_RNA_Primary.csv",
             "6K_RNA_Met.csv",
             "6K_RNA_ERPR_Primary.csv",
             "6K_RNA_ERPR_Met.csv",
             "6K_RNA_HER2_Primary.csv",
             "6K_RNA_HER2_Met.csv",
             "6K_RNA_TNBC_Primary.csv",
             "6K_RNA_TNBC_Met.csv",
             "6K_RNA_Under_40_Primary.csv",
             "6K_RNA_Under_40_Met.csv",
             "6K_RNA_Over_or_Equal_to_40_Primary.csv",
             "6K_RNA_Over_or_Equal_to_40_Met.csv",
]

#Where my csv files are stored on Kodiak
Data_directory = "/data/rentzc/Single_Cell_Spatial_Biology/Brm961b_022-6KPanel_COPY/groupedCopy/Breast_Core_Groups"

#Stating where outputs will be stored on Kodiak
Output_directory = "/data/rentzc/Single_Cell_Spatial_Biology/Project1_plots"
os.makedirs(Output_directory, exist_ok=True)

# Preprocessing using Scanpy
def General_UMAP(filepath, sample_name, Output_directory):
    print(f"Processing {sample_name}")

    # Loading CSV (rows are cells and columns are genes)
    df = pd.read_csv(filepath, index_col=0)

    # Convert all values to numeric (fixes dtype=object warning)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns (genes) with any NaNs after coercion
    df.dropna(axis=1, how='any', inplace=True)

    # Creating AnnData
    adata = sc.AnnData(df)

    # Ensure unique observation (cell) names
    adata.obs_names_make_unique()

    # Basic ScanPy Preprocessing
    sc.pp.filter_cells(adata, min_genes=200)  # gets rid of cells with fewer than 200 genes, default value
    sc.pp.filter_genes(adata, min_cells=3)    # gets rid of genes that are found in fewer than 3 cells, default value

    # can go back and make unique variants like mitochondrial genes

    # normalization step
    sc.pp.normalize_total(adata, target_sum=1e4)  # normalizes every cell to 10,000 UMI
    sc.pp.log1p(adata)                            # changes counts from UMI to log

    # start of clustering
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # default values

    # save raw data before processing further
    adata.raw = adata

    # subsetting adata to highly variable genes
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)  # default value

    # Calculate PCA and dimensionality reduction
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca_variance_ratio(adata, log=True)

    # computing neighbors and UMAP creation
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)  # default variables for now
    sc.tl.umap(adata)  # calculating the UMAP

    # Using Leiden algorithm to create clusters
    sc.tl.leiden(adata)
    sc.tl.leiden(adata, resolution=0.5, flavor="igraph", directed=False, n_iterations=2)  # default value so may change

    # Plot with Labels
    fig = sc.pl.umap(adata, color=['leiden'], show=False, return_fig=True)

    # remove _ and .csv from Plot label
    cleaned_name = sample_name.replace('_', ' ')  # '6K RNA HER2 Primary'
    ax = fig.axes[0]
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'Leiden Clustering: {cleaned_name}')
    ax.grid(False)

    # Saving plots to the Output_directory
    output_path = os.path.join(Output_directory, f"{sample_name}_umap.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved UMAP to {output_path}")

    # Find marker genes for these clusters created
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)  # plot top 25 genes in each cluster


# Run for each file
for file_name in csv_files:
    filepath = os.path.join(Data_directory, file_name)
    sample_name = os.path.splitext(file_name)[0]  # removes ".csv"

    if not os.path.isfile(filepath):
        print(f" File not found: {file_name}")
        continue

    try:
        General_UMAP(filepath, sample_name, Output_directory)
    except Exception as e:
        print(f" Error processing {file_name}: {e}")
        traceback.print_exc()

print("\n All processing complete.")


"""
TO DO: 
1) Use Scanpy to filter genes and pre-process
2) define cell types by markers 
2) define clusters by genes present
3) make separate UMAP plots of the following files:
--> All Primary = 6K_RNA_Primary.csv
--> All Secondary = 6K_RNA_Met.csv
--> HR+ primary = 6K_RNA_ERPR_Primary.csv  
--> HR+ secondary = 6K_RNA_ERPR_Met.csv    
--> HER2+ primary = 6K_RNA_HER2_Primary.csv   
--> HER2+ secondary = 6K_RNA_HER2_Met.csv   
--> TNBC primary = 6K_RNA_TNBC_Primary.csv
--> TNBC secondary = 6K_RNA_TNBC_Met.csv
--> <40 yos primary = 6K_RNA_Under_40_Primary.csv
--> <40 yos secondary = 6K_RNA_Under_40_Met.csv
--> >40 yos primary = 6K_RNA_Over_or_Equal_to_40_Primary.csv
--> >40 yos secondary = 6K_RNA_Over_or_Equal_to_40_Met.csv
"""