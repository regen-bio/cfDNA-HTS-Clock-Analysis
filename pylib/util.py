#!/usr/bin/env python3

import os
import shutil

from . import logger


DEFAULT_CACHE_PREFIX = os.path.join(os.path.dirname(__file__),
	"..", "..", ".cache_tmp")


def try_load_cache(fname: str, cache_prefix: str = None, *,
	auto_cache_if_missing: bool = False,
) -> str:
	return fname


def show_dict_tree(d: dict, indent: str = "\t", _level: int = 0):
	for k in sorted(d.keys()):
		# displayed version of k
		if isinstance(k, str):
			_k = f"\"{k}\""
		else:
			_k = str(k)
		# print per value type
		v = d[k]
		if isinstance(v, dict):
			print((indent * _level) + ("%s: " % _k))
			show_dict_tree(v, indent=indent, _level=_level + 1)
		else:
			print((indent * _level) + ("%s: %s" % (_k, type(v).__name__)))
	return


def merge_dict_inplace(merge_to: dict, merge_from: dict, *,
	overwrite: bool = False
) -> dict:
	# merge d2 into d1
	for k, v in merge_from.items():
		if k not in merge_to:
			merge_to[k] = v
		else:
			if isinstance(v, dict) and isinstance(merge_to[k], dict):
				merge_dict_inplace(merge_to[k], v)
			elif overwrite:
				merge_to[k] = v
			else:
				raise ValueError(f"key {k} exists in both merge_to "
					f"(as {type(merge_to[k]).__name__}) and merge_from "
					f"(as {type(v).__name__}), don't know how to merge")
	return merge_to
