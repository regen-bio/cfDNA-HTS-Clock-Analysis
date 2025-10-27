#!/usr/bin/env python3
# generating age prediction profiles by randomly generating beta values with
# beta-binomial model, fitted from the observed beta difference between
# replicates

import gzip
import os
import pdb
import pickle
import multiprocessing
import os
from typing import Sequence

import numpy
import pandas
import pyaging
import tqdm

import pylib

# force using cpu to avoid cuda memory issue
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class PredDataset(pylib.dataset.BetaDepth):
	@classmethod
	def load_dataset(cls, dataset: str):
		new = super().load_dataset(dataset, with_metadata=True,
			with_repl_group=True)
		new._post_load_preprocess()
		return new

	def _post_load_preprocess(self):
		pylib.logger.debug("post-load preprocess")
		beta = self.beta.T.reindex(self.metadata.index)
		# concate with metadata
		beta = pandas.concat([self.metadata, beta], axis=1)
		# epicv2 probe aggregation
		beta = pyaging.pp.epicv2_probe_aggregation(beta, verbose=False)
		# convert to adata
		adata = pyaging.pp.df_to_adata(beta,
			metadata_cols=self.metadata.columns,
			imputer_strategy="knn",
			verbose=False,
		)
		self.beta = adata.to_df()
		return


def _calc_repl_mean(beta: pandas.DataFrame, repl_to_subj: dict[str, str]
) -> pandas.DataFrame:
	pylib.logger.debug("calculating replicate mean")
	vals = beta.groupby(repl_to_subj).mean()
	ret = vals.loc[pandas.Series(repl_to_subj)].set_index(beta.index)
	return ret


def _calc_repl_std(beta: pandas.DataFrame, repl_to_subj: dict[str, str]
) -> pandas.DataFrame:
	pylib.logger.debug("calculating replicate std")
	vals = beta.groupby(repl_to_subj).std(ddof=1)
	ret = vals.loc[pandas.Series(repl_to_subj)].set_index(beta.index)
	return ret


def _predict_by_batch(n: int, beta: pandas.DataFrame, batch_size: int,
	clock: str,
) -> pandas.DataFrame:
	# create block data of the size as <batch_size> repeats of beta data, fill
	# in with random beta values and run predict_age
	# return the predictions, reshaped as n_sample * batch_size
	pylib.logger.debug(f"predicting by batch: size={batch_size}")
	n_sample, n_feat = beta.shape
	rnd_beta = numpy.empty((n_sample * batch_size, n_feat), dtype=numpy.float32)
	for i in range(0, n_sample * batch_size, n_sample):
		rnd_beta[i: i + n_sample] = (numpy.random.binomial(n, beta) / n).clip(0, 1)
	pylib.logger.debug(f"rnd_beta shape: {rnd_beta.shape}")
	# create pandas dataframe
	index = list()
	for i in range(batch_size):
		index.extend([v + ("_B%06d" % i) for v in beta.index])
	df = pandas.DataFrame(rnd_beta, index=index, columns=beta.columns)
	# transform into adata
	adata = pyaging.pp.df_to_adata(df, verbose=False)

	# predicting
	pyaging_data = pylib.util.try_load_cache("../pyaging_data",
		cache_prefix=pylib.util.DEFAULT_CACHE_PREFIX)
	pyaging.pred.predict_age(adata, [clock], dir=pyaging_data, verbose=False)

	# reshape predict data
	pred_vals = adata.obs[clock].values.reshape(n_sample, batch_size, order="F")
	pred_df = pandas.DataFrame(pred_vals, index=beta.index)
	return pred_df


def _predict_by_batch_worker(args: tuple) -> pandas.DataFrame:
	return _predict_by_batch(*args)


def _run_binomial_stoc_pred_per_clock(data: PredDataset, *,
	n_list: Sequence[int], clock: str, features: list[str],
	n_repeats: int = 1000, pool: multiprocessing.pool.Pool = None,
) -> dict[int, pandas.DataFrame]:

	# reduce beta to clock features
	beta = data.beta.reindex(columns=features)
	# do this to deal with those nans
	beta = pyaging.pp.df_to_adata(beta, verbose=False).to_df()

	beta_mean = _calc_repl_mean(beta, repl_to_subj=data.repl_to_subj)
	# beta_std = _calc_repl_std(beta, repl_to_subj=data.repl_to_subj)

	if pool is None:
		batch_size = 50
	else:
		batch_size = (n_repeats + pool._processes - 1) // pool._processes

	ret = dict()
	for n in n_list:
		pylib.logger.info(f"running: clock={clock}, n={n}")

		# batch args
		args_list = [(n, beta_mean, min(batch_size, n_repeats - i), clock)
			for i in range(0, n_repeats, batch_size)]

		if pool is not None:
			pylib.logger.info(f"running w/ {pool._processes} processes")
			batch_res_list = pool.map(_predict_by_batch_worker, args_list)
		else:
			pylib.logger.info("running w/o multiprocessing")
			batch_res_list = list(map(_predict_by_batch_worker, tqdm.tqdm(args_list)))

		# concat into a single prediction matrix of size n_samples * n_repeats
		df = pandas.concat([data.metadata] + batch_res_list, axis=1)
		df.columns = data.metadata.columns.tolist() + list(range(n_repeats))
		ret[n] = df

	return ret


def _run_beta_binomial_stoc_pred(dataset: str, *,
	n_list: Sequence[int], clocks: list[str], clock_cpgs: dict[str, list[str]],
	n_repeats: int = 1000, pool: multiprocessing.pool.Pool = None,
) -> dict[str, dict[int, pandas.DataFrame]]:
	data = PredDataset.load_dataset(dataset)

	res = dict()
	for clock in clocks:
		res[clock] = _run_binomial_stoc_pred_per_clock(data, n_list=n_list,
			clock=clock, features=clock_cpgs[clock], n_repeats=n_repeats,
			pool=pool,
		)
	return res


def _save_dataset_result_tmp(fname: str,
	res: dict[str, dict[int, pandas.DataFrame]],
) -> str:
	pylib.logger.info(f"saving dataset result (tmp): {fname}")
	with gzip.open(fname, "wb") as fp:
		pickle.dump(res, fp)
	return fname


def _merge_dataset_results(ofname: str, split_res_files: dict[str, str]) -> None:
	merged_res = dict()
	for ds, res_file in split_res_files.items():
		pylib.logger.info(f"merging dataset result: {ds} from {res_file}")
		with gzip.open(res_file, "rb") as fp:
			res = pickle.load(fp)
		merged_res[ds] = res
	# save merged results
	pylib.logger.info(f"saving merged result: {ofname}")
	with gzip.open(ofname, "wb") as fp:
		pickle.dump(merged_res, fp)
	return


def run_binom_stoc_pred_sim(*,
	output_file: str,
	datasets: str,
	n_list: Sequence[int],
	n_repeats: int = 1000,
	n_jobs: int = 16,
	log_level: int = pylib.log.logging.INFO,
	tmp_dir: str = ".tmp/",
):
	pylib.logger.setLevel(log_level)

	# check output file
	if os.path.exists(output_file):
		pylib.logger.error(f"output file exists: {output_file}")
		pylib.logger.error(f"aborting")
		return

	# load data
	clocks = pylib.clock.load_clocks()
	clock_cpgs = pylib.clock.load_clock_features()

	# enable parallelisation
	pool = multiprocessing.Pool(processes=n_jobs)

	# run prediction
	output_basename = os.path.basename(output_file)
	split_res_files = dict()
	for ds in datasets:
		ds_res = _run_beta_binomial_stoc_pred(ds, n_list=n_list, clocks=clocks,
			clock_cpgs=clock_cpgs, n_repeats=n_repeats, pool=pool,
		)
		res_fname = os.path.join(tmp_dir, f"{ds}.{output_basename}")
		split_res_files[ds] = _save_dataset_result_tmp(res_fname, ds_res)

	# merge results splits from each dataset into one file
	_merge_dataset_results(output_file, split_res_files)
	return


if __name__ == "__main__":
	run_binom_stoc_pred_sim(
		output_file="binomial_stoc.pred.pkl.gz",
		datasets=["RB_GALAXY", "RB_TWIST", "RB_GDNA_GALAXY", "RB_GDNA_TWIST"],
		n_list=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70,
			80, 90, 100, 120, 140, 160, 180, 200],
		n_repeats=10000,
		n_jobs=16,
	)
