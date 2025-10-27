#!/usr/bin/env python3

import pandas
import pyaging


raw = pandas.read_csv("output/shrink_table.tsv", sep="\t", index_col=0)

beta = pyaging.pp.epicv2_probe_aggregation(raw)

adata = pyaging.pp.df_to_adata(beta, imputer_strategy="knn")

beta = adata.to_df()

beta.to_csv("output/train_beta.tsv", sep="\t", float_format="%.4f", index=True)
