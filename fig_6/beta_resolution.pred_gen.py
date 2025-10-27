#!/usr/bin/env python3
# generating age prediction profiles by rouding the true beta values to its
# closest rational with denominator n

import functools
import gzip
import os
import pdb
import pickle
import multiprocessing
import multiprocessing.pool
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
	def load_dataset(cls, dataset: str) -> "PredDataset":
		return super().load_dataset(dataset, with_metadata=True,
			with_repl_group=True)

	@functools.cached_property
	def adata(self):
		pylib.logger.info("preparing adata")
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
		return adata


def _round_beta_by_n(beta: numpy.ndarray, n: int) -> numpy.ndarray:
	pylib.logger.info(f"rounding beta by n={n}")
	# round beta values to closest rational with denominator n
	# i.e. round to closest value in {0, 1/n, 2/n, ..., 1}
	return numpy.round(beta * n) / n


def _run_depth_round_pred_per_n(adata, n: int, clocks: list[str],
) -> pandas.DataFrame:
	adata = adata.copy()
	adata.X = _round_beta_by_n(adata.X, n)

	pylib.logger.info("predicting age")
	pyaging_data = pylib.util.try_load_cache("../pyaging_data",
		cache_prefix=pylib.util.DEFAULT_CACHE_PREFIX)
	pyaging.pred.predict_age(adata, clocks, dir=pyaging_data, verbose=False)
	return adata.obs


def _run_depth_round_pred(dataset: str, *,
	n_list: Sequence[int], clocks: list[str],
	pool: multiprocessing.pool.Pool = None,
) -> dict[int, pandas.DataFrame]:
	data = PredDataset.load_dataset(dataset)

	args_list = [(data.adata, n, clocks) for n in n_list]
	if pool is None:
		pylib.logger.info("running without parallelisation")
		res = [_run_depth_round_pred_per_n(*args) for args in
			tqdm.tqdm(args_list)]
	else:
		pylib.logger.info(f"running with {pool._processes} processes")
		res = pool.starmap(_run_depth_round_pred_per_n, args_list)

	# convert to dict
	ret = {a[1]: r for a, r in zip(args_list, res)}
	return ret


def _save(fname: str, res: dict[str, dict[int, pandas.DataFrame]],
) -> str:
	pylib.logger.info(f"saving to: {fname}")
	with gzip.open(fname, "wb") as fp:
		pickle.dump(res, fp)
	return fname


def run_depth_round_pred_sim(*,
	output_file: str,
	datasets: str,
	clocks: Sequence[str] = None,
	n_list: Sequence[int] = None,
	n_jobs: int = 8,
	log_level: int = pylib.log.logging.INFO,
):
	pylib.logger.setLevel(log_level)

	# check output file
	if os.path.exists(output_file):
		pylib.logger.error(f"output file exists: {output_file}")
		pylib.logger.error(f"aborting")
		return

	# prep parameters
	if clocks is None:
		clocks = pylib.clock.load_clocks()
	if n_list is None:
		n_list = [1, 2, 5, 10, 20, 50, 100]

	# run prediction with parallelisation
	res = dict()
	with multiprocessing.Pool(processes=n_jobs) as pool:
		# run prediction
		for ds in datasets:
			res[ds] = _run_depth_round_pred(ds, n_list=n_list,
				clocks=clocks, pool=pool,
			)

	# save results
	_save(output_file, res)
	return


if __name__ == "__main__":
	run_depth_round_pred_sim(
		output_file="beta_resolution.pred.pkl.gz",
		datasets=["RB_GALAXY", "RB_TWIST", "RB_GDNA_GALAXY", "RB_GDNA_TWIST"],
		n_list=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70,
			80, 90, 100, 120, 140, 160, 180, 200],
		n_jobs=8,
	)
