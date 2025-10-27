#!/usr/bin/env python3

import itertools
import json
import os
from typing import NoReturn

import matplotlib
import matplotlib.colors
import numpy
import pandas


def _read_color_cfg_csv(fname: str = None) -> pandas.DataFrame:
	if fname is None:
		fname = os.path.join(os.path.dirname(__file__), "data", "clock_cfg.csv")
	return pandas.read_csv(fname, sep=",")


def _assign_color_inplace(data: list[dict], colormap: str, *, seed: int = 42,
) -> NoReturn:
	n_colors = len(data)

	# get color list
	cmap = matplotlib.colormaps[colormap]
	numpy.random.seed(seed)
	color_i = numpy.random.permutation(numpy.linspace(0, 1, n_colors))

	for ci, d in zip(color_i, data):
		d["color"] = matplotlib.colors.to_hex(cmap(ci))
	return


def _add_marker_inplace(data: list[dict], markers: list[str]) -> NoReturn:
	for c, m in zip(data, itertools.cycle(markers)):
		c["marker"] = m
	return


def _save_clock_cfg_json(data: list[dict], fname: str = None) -> NoReturn:
	if fname is None:
		fname = os.path.join(os.path.dirname(__file__), "data", "clock_cfg.json")
	# transform data to dict of dicts, using v.name as key
	final_data = {v["name"]: v for v in data}
	with open(fname, "w") as fp:
		json.dump(final_data, fp, indent="\t", sort_keys=True)
	return


def _main():
	df = _read_color_cfg_csv()
	markers = ["o", "s", "D", "^", "v", "<", ">", "p", "h"]
	data = df.to_dict(orient="records")
	_assign_color_inplace(data, "turbo", seed=29)
	_add_marker_inplace(data, markers)
	_save_clock_cfg_json(data)
	return


if __name__ == "__main__":
	_main()
