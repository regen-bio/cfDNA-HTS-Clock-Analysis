#!/usr/bin/env python3

import dataclasses
import functools
import os
import pdb
import pickle

import numpy
import pandas
import pyaging
import sklearn
import sklearn.linear_model
import sklearn.model_selection

import pylib


@dataclasses.dataclass
class Dataset(pylib.dataset.BetaDepth):
	@functools.cached_property
	def adata(self):
		beta = self.beta.T.reindex(self.metadata.index)
		beta = pandas.concat([self.metadata, beta], axis=1)
		beta = pyaging.pp.epicv2_probe_aggregation(beta, verbose=False)
		adata = pyaging.pp.df_to_adata(beta,
			metadata_cols=self.metadata.columns,
			imputer_strategy="knn",
			verbose=False,
		)
		return adata


def _elasticnet_train(x, y, **kw) -> sklearn.linear_model.ElasticNetCV:
	en = sklearn.linear_model.ElasticNetCV(**kw)
	en.fit(x, y)
	return en


def _clock_no_transfer(data: Dataset, clock: str, *,
	common_cpgs: list[str], pca: sklearn.decomposition.PCA, en_args: dict,
) -> tuple[list, pandas.DataFrame]:
	# prepare data
	x_src = data.adata.to_df()
	# applying pca
	x_src = x_src.reindex(columns=common_cpgs, fill_value=0)
	x_src = pandas.DataFrame(pca.transform(x_src.values),
		index=x_src.index,
		columns=[f"pc_{i + 1}" for i in range(pca.n_components_)]
	)
	y_true = data.adata.obs["age"]

	# elasticnet train/validate with logo
	en_list = list()
	cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
	valid_res = pandas.DataFrame(index=x_src.index, columns=[clock])
	for train_idx, valid_idx in cv.split(x_src, y_true):
		# get split data
		train_x = x_src.iloc[train_idx]
		train_y = y_true.iloc[train_idx]
		# train model
		en = _elasticnet_train(train_x, train_y, **en_args)
		en_list.append(en)
		# validation
		valid_x = x_src.iloc[valid_idx]
		valid_res.iloc[valid_idx, 0] = en.predict(valid_x)

	return en_list, valid_res


def _clock_transfer_distill(data: Dataset, clock: str, *,
	common_cpgs: list[str], pca: sklearn.decomposition.PCA, en_args: dict,
) -> tuple[list, pandas.DataFrame]:
	# predicting with <clock>
	pyaging.pred.predict_age(data.adata, [clock], verbose=False,
		dir=pylib.util.try_load_cache("../pyaging_data"))
	# add a column to x_src as the prediction results of <clock>
	x_src = data.adata.to_df()
	# applying pca
	x_src = x_src.reindex(columns=common_cpgs, fill_value=0)
	# pdb.set_trace()
	x_src = pandas.DataFrame(pca.transform(x_src.values),
		index=x_src.index,
		columns=[f"pc_{i + 1}" for i in range(pca.n_components_)]
	)
	x_src[f"_pred_{clock}"] = data.adata.obs[clock]
	assert "age" not in x_src.columns
	y_true = data.adata.obs["age"]

	# elasticnet train/validate with logo
	en_list = list()
	cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
	valid_res = pandas.DataFrame(index=x_src.index, columns=[clock])
	for train_idx, valid_idx in cv.split(x_src, y_true):
		# get split data
		train_x = x_src.iloc[train_idx]
		train_y = y_true.iloc[train_idx]
		# train model
		en = _elasticnet_train(train_x, train_y, **en_args)
		en_list.append(en)
		# validation
		valid_x = x_src.iloc[valid_idx]
		valid_res.iloc[valid_idx, 0] = en.predict(valid_x)

	return en_list, valid_res


def _clock_transfer_dispatch(data: Dataset, clock: str, method: str, *,
	common_cpgs: list[str], pca: sklearn.decomposition.PCA,
	en_args: dict | None = None,
) -> tuple[list, pandas.DataFrame]:
	if en_args is None:
		en_args = dict()

	# dispatch per method
	if method == "distill":
		_trans_meth = _clock_transfer_distill
	else:
		raise ValueError(f"unknown method: {method}")

	# run model
	en_list, pred = _trans_meth(data, clock, common_cpgs=common_cpgs, pca=pca,
		en_args=en_args)
	return en_list, pred


def _train_per_dataset_worker(dataset: str, clocks: list[str], meths: list[str],
	common_cpgs: list[str], pca: sklearn.decomposition.PCA, en_args: dict,
) -> dict[str, dict]:

	pylib.logger.info(f"loading dataset: {dataset}")
	data = Dataset.load_dataset(dataset, with_metadata=True, with_repl_group=True)
	ret = dict()
	for meth in meths:
		pylib.logger.info(f"running transfer learning method: {meth}")
		# store results
		ens = dict()
		preds = list()
		# add no transfer learning as baseline
		en_list, pred = _clock_no_transfer(data, "no_transfer",
			common_cpgs=common_cpgs, pca=pca, en_args=en_args)
		preds.append(pred)
		# add transfer learning results
		for clock in clocks:
			pylib.logger.info(f"transfer clock: {clock}")
			en_list, pred = _clock_transfer_dispatch(data, clock, meth,
				common_cpgs=common_cpgs, pca=pca, en_args=en_args)
			ens[clock] = en_list
			preds.append(pred)

		ret[meth] = {
			# to record all elastic net models requires a little space ...
			"ens": ens,
			"preds": pandas.concat([data.metadata] + preds, axis=1),
		}

	return ret


def _dict_deep_merge_inplace(dest: dict, merge: dict) -> None:
	for key, value in merge.items():
		if isinstance(value, dict) and (key in dest):
			if isinstance(dest[key], dict):
				_dict_deep_merge_inplace(dest[key], value)
			else:
				raise ValueError("expect to merge dict into target dict, "
					f"but target is '{type(dest[key]).__name__}'")
		else:
			dest[key] = value
	return


def _save_results(fname: str, results: dict, *, merge: bool = False) -> None:
	if os.path.isfile(fname):  # exists
		if merge:
			# load old results
			pylib.logger.info(f"merging results with file '{fname}'")
			with open(fname, "rb") as fp:
				_res = pickle.load(fp)
			_dict_deep_merge_inplace(_res, results)
			results = _res
		else:
			pylib.logger.warning(f"file '{fname}' exists, will be overwritten "
				f"with merge={merge}")
	# save
	pylib.logger.info(f"saving file: {fname}")
	with open(fname, "wb") as fp:
		pickle.dump(results, fp)
	return


def _main():
	# configs
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	en_args = {
		"l1_ratio": (1 - numpy.logspace(-1, 0, 20) / 0.9),
		# "alphas": numpy.logspace(-3, 1, 30),
		"cv": 5,
		"max_iter": 10000,
		"n_jobs": 16,
		"verbose": False,
	}
	clocks = pylib.clock.load_clocks(exclude={"altumage"})
	meths = ["distill"]

	pylib.logger.info("clocks: %s" % clocks)
	pylib.logger.info("transfer learning methods: %s" % meths)

	# load data
	pylib.logger.info("loading pca")
	common_cpgs = pylib.pca.load_pca_cpgs()
	n_components_list, pca_models = pylib.pca.load_pca_models()
	pca = pca_models[max(n_components_list)]

	args_list = [(ds, clocks, meths, common_cpgs, pca, en_args) for ds in datasets]
	ds_res = [_train_per_dataset_worker(*args) for args in args_list]

	# save results
	res = dict(zip(datasets, ds_res))
	_save_results(f"transfer_learn.age_pred.pkl", results=res,
		merge=False)

	return


if __name__ == "__main__":
	_main()
