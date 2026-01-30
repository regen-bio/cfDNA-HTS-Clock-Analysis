#!/usr/bin/env python3

import argparse
import multiprocessing.pool
import os
import pdb
import pickle
from typing import Self, Sequence
import warnings

import numpy
import pandas
import pyaging
import sklearn
import sklearn.decomposition
import sklearn.exceptions
import sklearn.linear_model
import tqdm

import pylib


class CommaSepListAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, values.split(","))
		return


def get_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser()
	ap.add_argument("--train", action=CommaSepListAction, required=True,
		metavar="dataset[,dataset[,...]]",
		help="comma-separated training dataset names (required)",
	)
	ap.add_argument("--test", action=CommaSepListAction, required=True,
		metavar="dataset[,dataset[,...]]",
		help="comma-separated testing dataset names (required)",
	)
	ap.add_argument("--with-imputation", action="store_true",
		help="apply 0/1 imputation preprocessing strategy [no]",
	)
	ap.add_argument("-O", "--output-prefix", type=str, default="",
		metavar="prefix",
		help="output prefix [%(default)s]",
	)

	args = ap.parse_args()

	return args


class Dataset(pylib.dataset.FinalBetaOnly):
	@classmethod
	def load_dataset(cls, *ka, with_metadata: bool = True, **kw) -> Self:
		new = super().load_dataset(*ka, with_metadata=with_metadata, **kw)
		# drop samples with nan age
		valid_index = new.metadata["age"].dropna().index
		new.metadata = new.metadata.reindex(index=valid_index)
		new.beta = new.beta.reindex(columns=valid_index)
		return new

	@classmethod
	def load_multi(cls, dataset_list: Sequence[str], *ka,
		with_metadata: bool = True, metadata_policy: str = "minimal",
		**kw,
	) -> Self:
		pylib.logger.info(("loading concatenated multiple datasets: {}")
			.format(",".join(dataset_list)))
		if metadata_policy not in ["minimal", "full"]:
			raise ValueError("metadata policy must be one of ['minimal', 'full']")

		if not dataset_list:
			raise ValueError("dataset_list cannot be empty")

		# load each dataset and concatenate
		data_list = [cls.load_dataset(ds, *ka, with_metadata=with_metadata, **kw)
			for ds in dataset_list]

		# concat
		pylib.logger.info("concatenating")
		beta = pandas.concat([d.beta for d in data_list], axis=1)

		if with_metadata:
			if metadata_policy == "minimal":
				# the final columns are the intersection of all metadata columns
				cols = data_list[0].metadata.columns
				for d in data_list[1:]:
					cols = cols.intersection(d.metadata.columns)
				metadata = pandas.concat([d.metadata[cols] for d in data_list],
					axis=0)
			elif metadata_policy == "full":
				# reserve all columns, fill missing with nan
				metadata = pandas.concat([d.metadata for d in data_list],
					axis=0, join="outer")
		else:
			metadata = None

		return cls(beta=beta, metadata=metadata)

	@property
	def ill_mask(self) -> pandas.DataFrame:
		return (self.beta == 0) | (self.beta == 1)


def _get_pca_model():
	n_components_list, pca_models = pylib.pca.load_pca_models()
	pca = pca_models[max(n_components_list)]
	return pca


def _preprocess(data: Dataset, with_imputation: bool,
) -> pylib.dataset.DatasetBase:
	beta = data.beta.copy()

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
	pyaging_data = pylib.util.try_load_cache("pyaging_data")
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


def _elasticnet_train(x: numpy.ndarray, y: numpy.ndarray, **kw,
) -> tuple[sklearn.linear_model.ElasticNetCV, numpy.ndarray]:
	l1_ratio = (1 - numpy.logspace(-1, 0, 20) / 0.9)
	alphas = numpy.logspace(-3, 1, 30)
	en = sklearn.linear_model.ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
		max_iter=10000, cv=5, n_jobs=8, verbose=True, **kw)
	en.fit(x, y)
	pred = en.predict(x)
	return en, pred


def _distill_cross_test_train(*, dataset_list: Sequence[str],
	with_imputation: bool, pca: sklearn.decomposition.PCA,
	pca_cpgs: Sequence[str], clocks: Sequence[str],
) -> tuple[dict[str, sklearn.linear_model.ElasticNetCV], pandas.DataFrame]:
	# load training data
	pylib.logger.info("preprocessing training data")
	data = _preprocess(Dataset.load_multi(dataset_list),
		with_imputation=with_imputation,
	)
	pca_beta = _apply_pca(data.beta, pca, pca_cpgs)
	teacher_y = _predict_age_plain(data, models=clocks)

	# run in parallel
	pylib.logger.info("training")
	pool = multiprocessing.pool.ThreadPool(8)

	args_list = [
		(
			# train x
			numpy.hstack(
				[pca_beta.values.T, teacher_y[c].values.reshape(-1, 1)]),
			# train y
			data.metadata["age"].values,
		)
		for c in clocks
	]

	train_res = pool.starmap(_elasticnet_train, args_list)

	# combine train results
	en_dict = dict()
	train_pred = pandas.DataFrame(index=data.metadata.index, columns=clocks)
	for c, (en, pred) in zip(clocks, train_res):
		en_dict[c] = en
		train_pred[c] = pred

	return en_dict, train_pred


def _distill_cross_test_test(*, en_dict: dict[str, sklearn.linear_model.ElasticNetCV],
	dataset_name: str, with_imputation: bool, pca: sklearn.decomposition.PCA,
	pca_cpgs: Sequence[str],
) -> pandas.DataFrame:
	clocks = sorted(en_dict.keys())

	# load testing data
	pylib.logger.info("preprocessing testing data")
	data = _preprocess(Dataset.load_dataset(dataset_name),
		with_imputation=with_imputation,
	)
	pca_beta = _apply_pca(data.beta, pca, pca_cpgs)
	teacher_y = _predict_age_plain(data, models=clocks)

	# tesing for each clock
	cross_pred = pandas.DataFrame(index=data.metadata.index, columns=clocks)
	for c in clocks:
		en = en_dict[c]
		x_test = numpy.hstack(
			[pca_beta.values.T, teacher_y[c].values.reshape(-1, 1)])
		y_pred = en.predict(x_test)
		cross_pred[c] = y_pred

	# concat metadata
	base_pred = pandas.concat([data.metadata, teacher_y], axis=1)
	cross_pred = pandas.concat([data.metadata, cross_pred], axis=1)

	return base_pred, cross_pred


def _run_distill_cross_test(*, train_dataset_list: Sequence[str],
	test_dataset_list: Sequence[str], with_imputation: bool = False,
	output_prefix: str = "",
) -> None:
	# load and preprocess
	pca_cpgs = pylib.pca.load_pca_cpgs()
	pca = _get_pca_model()
	clocks = pylib.clock.load_clocks(exclude={"altumage"})

	# train
	en_dict, train_pred = _distill_cross_test_train(
		dataset_list=train_dataset_list,
		with_imputation=with_imputation,
		pca=pca, pca_cpgs=pca_cpgs,
		clocks=clocks,
	)

	# test at per dataset basis
	for test_dataset_name in tqdm.tqdm(test_dataset_list):
		pylib.logger.info(
			f"testing on dataset {test_dataset_name} using trained models")
		base_pred, cross_pred = _distill_cross_test_test(
			en_dict=en_dict,
			dataset_name=test_dataset_name,
			with_imputation=with_imputation,
			pca=pca, pca_cpgs=pca_cpgs,
		)

		# oragnize results
		pylib.logger.info("collecting results")
		res = {
			"train_list": train_dataset_list,
			"test": test_dataset_name,
			"with_imputation": with_imputation,
			"en_dict": en_dict,
			"base_pred": base_pred,
			"cross_pred": cross_pred,
		}

		# save results
		train_str = ("-").join(train_dataset_list)
		imput_str = "with_imput" if with_imputation else "no_imput"
		if (output_prefix == "") or output_prefix.endswith(os.path.sep):
			output_prefix += "."
		output_fname = (f"{output_prefix}.{train_str}."
			f"{test_dataset_name}.{imput_str}.age_pred.pkl")
		pylib.logger.info(f"saving as {output_fname}")
		with open(output_fname, "wb") as f:
			pickle.dump(res, f)

	return


def _main():
	args = get_args()
	warnings.filterwarnings("ignore",
		category=sklearn.exceptions.ConvergenceWarning)

	_run_distill_cross_test(
		train_dataset_list=args.train,
		test_dataset_list=args.test,
		with_imputation=args.with_imputation,
		output_prefix=args.output_prefix,
	)
	return


if __name__ == "__main__":
	_main()
