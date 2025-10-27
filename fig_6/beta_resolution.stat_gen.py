#!/usr/bin/env python3

import dataclasses
import gzip
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


def _prep_depth_round_res(fname: str, clocks: list[str],
	series_cfgs: list[SeriesCfg],
) -> dict[str, dict[str, pandas.DataFrame]]:
	# load data
	pylib.logger.info(f"loading data from {fname}")
	with gzip.open(fname, "rb") as fp:
		data = pickle.load(fp)

	# calculate
	ret = dict()
	for ds, ds_res in data.items():
		pylib.logger.info(f"processing dataset: {ds}")
		n_list = sorted(ds_res.keys())

		# load external data
		repl_group = pylib.dataset.ReplGroup.load_dataset(ds)

		# calculate stats
		ds_stat = dict()
		for cfg in series_cfgs:
			ds_stat[cfg.key] = pandas.DataFrame(
				index=n_list, columns=clocks, dtype=float)

		for n, df in tqdm.tqdm(ds_res.items(), desc=f"processing {ds}"):
			_stat = pylib.AgePredRes(df).calc_stat(clocks, repl_group=repl_group)
			for cfg in series_cfgs:
				ds_stat[cfg.key].loc[n, :] = _stat[cfg.key]

		ret[ds] = ds_stat

	return ret


def _main():
	# configs
	clocks = pylib.clock.load_clocks()
	series_cfgs = [
		SeriesCfg(key="mae", color="#3F6EE5", display_name="MAE", unit="years"),
		SeriesCfg(key="repl_mad", color="#E5763F", display_name="RD", unit="years"),
	]

	stat = _prep_depth_round_res(
		fname="beta_resolution.pred.pkl.gz",
		clocks=clocks, series_cfgs=series_cfgs
	)

	with open("beta_resolution.stat.pkl", "wb") as fp:
		pickle.dump(stat, fp)

	return


if __name__ == "__main__":
	_main()
