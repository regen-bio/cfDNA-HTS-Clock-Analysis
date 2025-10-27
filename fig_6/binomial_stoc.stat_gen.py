#!/usr/bin/env python3

import dataclasses
import gzip
import itertools
import multiprocessing
import multiprocessing.pool
import pdb
import pickle

import pandas
import tqdm

import pylib


@dataclasses.dataclass
class SeriesCfg(object):
	key: str
	color: str
	display_name: str
	unit: str


def _load_binomial_stoc_results(fname: str
) -> dict[str, dict[str, dict[int, pandas.DataFrame]]]:
	pylib.logger.info(f"loading data from {fname}")
	with gzip.open(fname, "rb") as fp:
		ret = pickle.load(fp)
	return ret


def _stat_worker(df: pandas.DataFrame, repl_group: pylib.dataset.ReplGroup,
	n: int, clock: str
) -> tuple[int, str, pandas.DataFrame]:
	age_pred_res = pylib.age_pred_res.AgePredRes(df)
	data_columns = list(range(df.shape[1] - 3))
	stat = age_pred_res.calc_stat(data_columns, repl_group=repl_group)
	return n, clock, stat


def _condense_stat(data: dict[str, dict[str, dict[int, pandas.DataFrame]]],
	series_cfgs: list[SeriesCfg], *, pool: multiprocessing.pool.Pool,
) -> dict[str, dict[str, pandas.DataFrame]]:
	ret = dict()

	pylib.logger.info("condensing statistics")

	for ds, ds_res in data.items():
		# find clocks here
		clocks = sorted(ds_res.keys())
		assert len(clocks) > 0
		n_list = sorted(ds_res[clocks[0]].keys())

		repl_group = pylib.dataset.ReplGroup.load_dataset(ds)

		args_list = []
		for clock, n in itertools.product(clocks, n_list):
			args_list.append((ds_res[clock][n], repl_group, n, clock))

		# run in parallel
		pylib.logger.info(f"condensing with {pool._processes} processes")
		_res = pool.starmap(_stat_worker, tqdm.tqdm(args_list))

		# collect results
		pylib.logger.info("collecting results from condensed stats")
		condens = {cfg.key: pandas.DataFrame(index=n_list, columns=clocks)
			for cfg in series_cfgs}

		for n, clock, age_pred_stat in _res:
			for cfg in series_cfgs:
				condens[cfg.key].at[n, clock] = age_pred_stat[cfg.key].mean()

		# add to ret
		ret[ds] = condens

	return ret


def _main():
	# configs
	series_cfgs = [
		SeriesCfg(key="mae", color="#3F6EE5", display_name="MAE", unit="years"),
		SeriesCfg(key="repl_mad", color="#E5763F", display_name="RD", unit="years"),
	]

	# load data
	data = _load_binomial_stoc_results(
		"binomial_stoc.pred.pkl.gz"
	)

	pool = multiprocessing.Pool(processes=16)

	# condense statistics
	stat = _condense_stat(data, series_cfgs, pool=pool)

	# save results
	fname = "binomial_stoc.stat.pkl"
	pylib.logger.info(f"save results: {fname}")
	with open(fname, "wb") as fp:
		pickle.dump(stat, fp)
	return


if __name__ == "__main__":
	_main()
