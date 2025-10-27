#!/usr/bin/env python3

import gzip
import os
import pickle

import pandas
import pyaging
import sklearn
import sklearn.linear_model
import sklearn.metrics

import pylib


def open_any(*ka, func=open, open_args=None, araise=True):
	if open_args is None:
		open_args = dict()
	for fname in ka:
		if os.path.exists(fname):
			return func(fname, **open_args)
	if araise:
		files = (",").join(ka)
		raise FileNotFoundError(f"none of the files '{files}' exist.")
	return None


def prep_beta(dataset: pylib.dataset.DatasetBase, cpgs: list[str],
) -> pandas.DataFrame:
	beta = dataset.beta
	# filter cpgs
	beta = beta.reindex(cpgs, fill_value=0)
	# transpose
	beta = beta.T
	# fill scattered nans using pyaging
	beta = pyaging.pp.epicv2_probe_aggregation(beta)
	adata = pyaging.pp.df_to_adata(beta, metadata_cols=list(),
		imputer_strategy="knn")
	beta = adata.to_df()
	# last time reconcile cpgs
	beta = beta.T.reindex(cpgs, fill_value=0).T
	return beta


def load_trained_models(fname: str) -> list[sklearn.linear_model.ElasticNet]:
	with gzip.open(fname, "rb") as fp:
		ret = pickle.load(fp)
	return ret


def predict_rb_data(model_train_res: list[sklearn.linear_model.ElasticNet],
	beta: pandas.DataFrame, age: pandas.Series,
) -> pandas.DataFrame:
	ret = pandas.DataFrame(
		index=range(len(model_train_res)),
		columns=["alpha", "l1_ratio", "n_cpgs", "train_mae", "train_rmse",
			"pred_mae", "pred_rmse", "n_iter"] + beta.index.tolist(),
	)
	beta_val = beta.values
	true_age = age.reindex(beta.index).values
	for i, train_res in enumerate(model_train_res):
		pylib.logger.info(f"predicting with model {i}")

		# predict
		pred_age = train_res["model"].predict(beta_val)
		train_mae = sklearn.metrics.mean_absolute_error(true_age, pred_age)
		train_rmse = sklearn.metrics.mean_squared_error(true_age, pred_age)

		# summarize results
		vals = [train_res[k] for k in ["alpha", "l1_ratio", "n_cpgs", "mae",
			"rmse"]]
		vals += [train_mae, train_rmse, train_res["model"].n_iter_]
		vals += pred_age.tolist()

		ret.loc[i] = vals
	return ret


def main():
	# configs
	datasets = ["RB_935K", "RB_MSA", "RB_GALAXY", "RB_TWIST",
		"RB_GDNA_GALAXY", "RB_GDNA_TWIST",
		"GSE83944", "GSE247193", "GSE247195", "GSE247197"]

	# load cpgs
	with open("../pca_gse55763/input/common_cpgs.list", "r") as fp:
		cpgs = fp.read().splitlines()

	# load predicting data
	rb_beta: dict[pandas.DataFrame] = dict()
	rb_age: dict[pandas.Series] = dict()

	for ds in datasets:
		pylib.logger.info(f"loading data for {ds}")
		dataset = pylib.dataset.BetaOnly.load_dataset(ds, with_metadata=True)
		rb_beta[ds] = prep_beta(dataset, cpgs)
		rb_age[ds] = dataset.metadata["age"]

	# load trained models
	models = load_trained_models("train_res/train_full.pkl.gz")

	# predict
	for ds in datasets:
		pylib.logger.info(f"predicting for {ds}")
		pred_df = predict_rb_data(models, rb_beta[ds], rb_age[ds])
		pylib.logger.info(f"saving result for {ds}")
		pred_df.to_csv(f"pred_res/reprod.{ds}.pred.tsv", sep="\t",
			index=False)

	return


if __name__ == "__main__":
	main()
