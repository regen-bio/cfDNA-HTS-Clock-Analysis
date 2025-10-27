#!/usr/bin/env python3

import collections
import functools
import json

import numpy
import threadpoolctl
import pandas

import pylib


class BetaDepth(pylib.dataset.BetaDepth):
	@functools.cached_property
	def mask(self) -> pandas.DataFrame:
		return (self.beta == 0) | (self.beta == 1)


def _est_beta_hist(data: BetaDepth, n_bins: int = 200,
) -> tuple[numpy.ndarray, numpy.ndarray]:
	# extract data
	mask = numpy.isfinite(data.beta.values)
	values = data.beta.values[mask]
	# calculate bin edges
	bin_w = 1.0 / (n_bins - 1)
	edges = numpy.linspace(0 - bin_w / 2, 1 + bin_w / 2, n_bins + 1)
	hist = numpy.histogram(values, bins=edges, density=True)[0]
	return edges, hist


def _stat_ill_cpg_depth_count(data: BetaDepth,
) -> tuple[float, collections.Counter]:
	ill_frac = data.mask.values.mean()
	# count ill cpgs at each depth
	ill_depths = data.depth.values[data.mask.values]
	ill_depth_count = collections.Counter(ill_depths)
	# turn key numpy.float64 into int
	ill_depth_count = collections.Counter({int(k): v
		for k, v in ill_depth_count.items()})
	return ill_frac, ill_depth_count


def _stat_clock_affected_frac(data: BetaDepth, clock_cpgs: dict[str, list],
) -> dict[str, float]:
	ret = dict()
	for clock, cpg_list in clock_cpgs.items():
		n_cpg = len(cpg_list)
		clock_mask = data.mask.reindex(index=cpg_list, fill_value=False)
		n_ill = clock_mask.values.sum(axis=0)
		assert len(n_ill) == clock_mask.shape[1]
		mean_frac = (n_ill / n_cpg).mean()
		ret[clock] = mean_frac

	return ret


def _run_ill_cpg_stat(dataset: str, clock_cpgs: dict[str, list]) -> dict:
	data = BetaDepth.load_dataset(dataset)
	# beta distr.
	edges, hist = _est_beta_hist(data, n_bins=100)
	# count depths of ill cpgs
	ill_frac, ill_depth_count = _stat_ill_cpg_depth_count(data)
	# calc per clock frac of affected cpgs
	clock_affected_frac = _stat_clock_affected_frac(data, clock_cpgs)

	ret = {
		"beta_hist_bins": edges.tolist(),
		"beta_hist": hist.tolist(),
		"ill_frac": ill_frac,
		"ill_depth_count": ill_depth_count,
		"clock_affected_frac": clock_affected_frac,
	}
	return ret


def _main():
	# configs
	datasets = ["RB_GALAXY", "RB_TWIST", "RB_GDNA_GALAXY", "RB_GDNA_TWIST",
		"RB_SYF", "GSE144691", "GSE86832", "BUCCAL_TWIST"]

	# load data
	clocks = pylib.clock.load_clocks()
	clock_cpgs = pylib.clock.load_clock_features(
		"../etc/pyaging_clock_cpgs.json")
	clock_cpgs = {c: clock_cpgs[c] for c in clocks}

	# run stats
	res = dict()
	for ds in datasets:
		res[ds] = _run_ill_cpg_stat(ds, clock_cpgs)

	# save results
	ofname = "ill_cpg_stat.json"
	pylib.logger.info(f"saving results: {ofname}")
	with open(ofname, "w") as fp:
		json.dump(res, fp)
	return


if __name__ == "__main__":
	with threadpoolctl.threadpool_limits(limits=24, user_api="blas"):
		_main()
