#!/usr/bin/env python3

import os
import pickle

import tqdm

from . import logger, util


def load_pca_cpgs(fname: str = None, *,
	cache_prefix: str = util.DEFAULT_CACHE_PREFIX,
) -> list[str]:
	if fname is None:
		fname = os.path.join(os.path.dirname(__file__),
			"data/pca_cpgs.txt")
	fname = util.try_load_cache(fname, cache_prefix=cache_prefix)
	with open(fname, "r") as fp:
		ret = fp.read().splitlines()
	return ret


def load_n_components_list(fname: str = None, *,
	cache_prefix: str = util.DEFAULT_CACHE_PREFIX,
) -> list[int]:
	if fname is None:
		fname = os.path.join(os.path.dirname(__file__),
			"data/pca_n_components.txt")
	fname = util.try_load_cache(fname, cache_prefix=cache_prefix)
	with open(fname, "r") as fp:
		ret = [int(x) for x in fp.read().splitlines()]
	return ret


def load_pca_models(dirname: str = "pca_res",
	n_components_list: list[int] = None, *,
	cache_prefix: str = util.DEFAULT_CACHE_PREFIX,
) -> dict[int]:
	if n_components_list is None:
		n_components_list = load_n_components_list()

	model_dict = dict()
	for n in tqdm.tqdm(n_components_list):
		fname = util.try_load_cache(
			os.path.join(dirname, f"train_beta_pca_{n}.pkl"),
			cache_prefix=cache_prefix,
		)
		with open(fname, "rb") as fp:
			model_dict[n] = pickle.load(fp)
	return n_components_list, model_dict
