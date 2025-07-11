"""
Author: Carolyn Rentz
Assignment Title: Project 1: Adding in genes
Date Created: 7/10/25
Date Last Modified: 7/10/25
"""

import traceback
import scanpy as sc
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import os as os

#Where my csv files are stored on Kodiak
Data_directory = "/data/rentzc/Single_Cell_Spatial_Biology/Brm961b_022-6KPanel_COPY/groupedCopy/Breast_Core_Groups"

#Stating where outputs will be stored on Kodiak
Output_directory = "/data/rentzc/Single_Cell_Spatial_Biology/Project1_plots"
os.makedirs(Output_directory, exist_ok=True)

#Attempt 1 CSV file first
input_file = os.path.join(Data_directory, "6K_RNA_Primary.csv")
df = pd.read_csv(input_file, index_col=0)

#create the AnnData Object
adata = sc.AnnData(df)

#Preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]  # keep only variable genes

#Dimensionality Reduction
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

# Compute neighborhood graph and cluster
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)  #may change res

#Marker Identification
sc.tl.rank_genes_groups(adata,'leiden', method='wilcoxon')
marker_genes_df = sc.get.rank_genes_groups_df(adata, group=None)




