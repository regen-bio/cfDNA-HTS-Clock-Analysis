#!/usr/bin/env python3

import json
import os
from typing import Sequence

from . import logger, util


def load_clocks(fname: str = None, exclude: Sequence[str] = None) -> list[str]:
	if fname is None:
		fname = os.path.join(os.path.dirname(__file__), "data", "clocks.txt")
	logger.info(f"loading clock list from '{fname}'")
	with open(fname, "r") as fp:
		ret = fp.read().strip().split(",")
	if exclude is not None:
		ret = [x for x in ret if x not in exclude]
	return ret


def load_clock_features(fname: str = None, *,
	cache_prefix: str = util.DEFAULT_CACHE_PREFIX
) -> dict[str, list[str]]:
	if fname is None:
		fname = util.try_load_cache("../etc/pyaging_clock_cpgs.json",
			cache_prefix=cache_prefix)
	logger.info(f"loading clock features from '{fname}'")
	with open(fname, "r") as fp:
		ret = json.load(fp)
	return ret
