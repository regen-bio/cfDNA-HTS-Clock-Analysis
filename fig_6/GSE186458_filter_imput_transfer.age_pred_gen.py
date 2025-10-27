#!/usr/bin/env python3

import itertools
import multiprocessing.pool
import os
import pdb
import pickle
from typing import NoReturn, Self, Sequence

import numpy
import pandas
import pyaging
import sklearn
import sklearn.linear_model
import sklearn.model_selection

import pylib


class Dataset(pylib.dataset.BetaDepth):
	@classmethod
	def load_dataset(cls, *ka, with_metadata: bool = True, **kw) -> Self:
		return super().load_dataset(*ka, with_metadata=with_metadata, **kw)

	@property
	def ill_mask(self) -> pandas.DataFrame:
		return (self.beta == 0) | (self.beta == 1)


def _load_clock_list(fname: str) -> list[str]:
	with open(fname, "r") as fp:
		ret = fp.read().splitlines()
	return ret


def _get_pca_model():
	n_components_list, pca_models = pylib.pca.load_pca_models()
	pca = pca_models[max(n_components_list)]
	return pca


def _preprocess(data: Dataset, *, with_depth_filtering: int = None,
	with_imputation: bool,
) -> pylib.dataset.DatasetBase:
	beta = data.beta.copy()

	if (with_depth_filtering is not None) and (with_depth_filtering > 0):
		beta = beta.where(data.depth >= with_depth_filtering, other=float("nan"))

	if with_imputation:
		beta = beta.where(~data.ill_mask, other=float("nan"))

	# make pyaging adata
	beta = beta.T.reindex(data.metadata.index)
	beta = pyaging.pp.epicv2_probe_aggregation(beta, verbose=False)

	ret = pylib.dataset.DatasetBase(beta=beta.T, depth=None,
		metadata=data.metadata)
	return ret


def _predict_age_plain(data: pylib.dataset.DatasetBase, models: Sequence[str],
) -> pandas.DataFrame:
	pylib.logger.info("making adata for pyaging prediction")
	adata = pyaging.pp.df_to_adata(data.beta.T,
		imputer_strategy="knn", verbose=False,
	)
	pylib.logger.info("predicting age with pyaging")
	pyaging_data = pylib.util.try_load_cache("../pyaging_data")
	pyaging.pred.predict_age(adata, models, dir=pyaging_data, verbose=False)
	return adata.obs


def _apply_pca(beta: pandas.DataFrame, pca: sklearn.decomposition.PCA,
	pca_cpgs: list[str]
) -> pandas.DataFrame:
	beta_pca = beta.reindex(pca_cpgs, fill_value=0).values.T
	assert beta_pca.shape[1] == len(pca_cpgs)
	# at this stage, has to fill all nans with 0
	numpy.nan_to_num(beta_pca, copy=False, nan=0.0)
	beta_pca = pca.transform(beta_pca)

	ret = pandas.DataFrame(beta_pca.T, columns=beta.columns,
		index=[f"pc_{i + 1}" for i in range(pca.n_components_)]
	)
	return ret


def _elasticnet_train(x, y, **kw) -> sklearn.linear_model.ElasticNetCV:
	l1_ratio = (1 - numpy.logspace(-1, 0, 20) / 0.9)
	alphas = numpy.logspace(-3, 1, 30)
	en = sklearn.linear_model.ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
		max_iter=10000, cv=5, n_jobs=8, verbose=True, **kw)
	en.fit(x, y)
	return en


def _predict_age_distill_single(x: numpy.ndarray, y: numpy.ndarray,
) -> numpy.ndarray:
	ret = numpy.empty(len(y), dtype=float)

	cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
	for train_idx, valid_idx in cv.split(x, y):
		# train elastic net with part data
		EN = _elasticnet_train(x[train_idx], y[train_idx])
		# evaluate with k-fold cv
		ret[valid_idx] = EN.predict(x[valid_idx])

	return ret


def _predict_age_distill(data: pylib.dataset.DatasetBase, models: Sequence[str],
	*, pca: sklearn.decomposition.PCA, pca_cpgs: Sequence[str],
	pool: multiprocessing.pool.Pool,
) -> pandas.DataFrame:
	# apply pca
	pca_beta = _apply_pca(data.beta, pca, pca_cpgs)
	distill_y = _predict_age_plain(data, models)

	# predicting in parallel
	args_list = [
		(
			numpy.hstack([pca_beta.values.T, distill_y[m].values.reshape(-1, 1)]),
			data.metadata["age"].values
		)
		for m in models
	]
	pred = pool.starmap(_predict_age_distill_single, args_list)

	# collect results
	ret = pandas.DataFrame(index=data.metadata.index, columns=models)
	for m, p in zip(models, pred):
		ret[m] = p

	return ret


def _predict_age(data: pylib.dataset.DatasetBase, models: Sequence[str],
	*, with_transfer_learning: bool, pca: sklearn.decomposition.PCA,
	pca_cpgs: Sequence[str], pool: multiprocessing.pool.Pool,
) -> pandas.DataFrame:
	if not with_transfer_learning:
		pred = _predict_age_plain(data, models)
	else:
		pred = _predict_age_distill(data, models, pca=pca, pca_cpgs=pca_cpgs,
			pool=pool)
	# add metadata
	ret = pandas.concat([data.metadata, pred], axis=1)
	return ret


def _save_result_tmp(res: pandas.DataFrame, *,
	with_depth_filtering: int,
	with_imputation: bool,
	with_transfer_learning: bool,
	clocks: Sequence[str],
) -> NoReturn:
	fname = os.path.join(".tmp",
		("GSE186458_filter_imput_transfer.age_pred.{}.{}.{}.pkl").format(
			with_depth_filtering,
			int(with_imputation or 0),
			int(with_transfer_learning or 0),
		)
	)
	key = (with_depth_filtering, with_imputation, with_transfer_learning)
	obj = {
		"key_fields": ("depth_filtering", "imputation", "transfer_learning"),
		"clocks": clocks,
		"res": {key: res},
	}
	pylib.logger.info(f"saving tmp result file: {fname}")
	with open(fname, "wb") as fp:
		pickle.dump(obj, fp)

	return


def _run_filter_imput_transfer(with_depth_filtering: int,
	with_imputation: bool,
	with_transfer_learning: bool,
) -> None:
	# load and preprocess
	pca_cpgs = pylib.pca.load_pca_cpgs()
	pca = _get_pca_model()
	clocks = pylib.clock.load_clocks(exclude={"altumage"})

	pool = multiprocessing.pool.ThreadPool(8)

	dataset = Dataset.load_dataset("GSE186458")
	prep_data = _preprocess(dataset,
		with_depth_filtering=with_depth_filtering,
		with_imputation=with_imputation,
	)

	# predict age
	_res = _predict_age(prep_data, clocks, pca=pca, pca_cpgs=pca_cpgs,
		with_transfer_learning=with_transfer_learning, pool=pool,
	)

	# save results
	_save_result_tmp(_res,
		with_depth_filtering=with_depth_filtering,
		with_imputation=with_imputation,
		with_transfer_learning=with_transfer_learning,
		clocks=clocks,
	)
	return


def _main():
	# run all combinations
	for with_depth_filtering, with_imputation, with_transfer_learning in itertools.product(
		[0, 20], [False, True], [False, True]
	):
		pylib.logger.info(
			f"running with_depth_filtering={with_depth_filtering}, "
			f"with_imputation={with_imputation}, "
			f"with_transfer_learning={with_transfer_learning}"
		)
		_run_filter_imput_transfer(
			with_depth_filtering=with_depth_filtering,
			with_imputation=with_imputation,
			with_transfer_learning=with_transfer_learning,
		)
	return


if __name__ == "__main__":
	_main()
