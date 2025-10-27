#!/usr/bin/env python3

import dataclasses
import functools
import json
import os
from typing import Self

import pandas

from . import logger, util


class ReplGroup(list[dict[str]]):
	@classmethod
	def load_dataset(cls, dataset: str, *,
		cache_prefix: str = util.DEFAULT_CACHE_PREFIX
	) -> Self:
		fname = util.try_load_cache(f"data/{dataset}/repl.json",
			cache_prefix=cache_prefix, auto_cache_if_missing=True)
		with open(fname, "r") as fp:
			data = json.load(fp)
		return cls(data)

	@functools.cached_property
	def repl_to_subj(self) -> dict[str, str]:
		ret = {r: g["subject_id"] for g in self for r in g["replicates"]}
		return ret


@dataclasses.dataclass
class DatasetBase(object):
	beta: pandas.DataFrame
	depth: pandas.DataFrame
	metadata: pandas.DataFrame = None
	repl_group: ReplGroup = None

	@property
	def repl_to_subj(self) -> dict[str, str]:
		if self.repl_group is None:
			raise ValueError("replicate group information is not available, "
				"reload the dataset with `with_repl_group=True`")
		return self.repl_group.repl_to_subj

	@staticmethod
	def _load_tsv(fname: str, cache_prefix: str = util.DEFAULT_CACHE_PREFIX
		) -> pandas.DataFrame:
		fname = util.try_load_cache(fname, cache_prefix=cache_prefix)
		logger.debug(f"loading tsv: {fname}")
		ret = pandas.read_csv(fname, sep="\t", index_col=0)
		if (dup_idx := ret.index.duplicated(keep="first")).any():
			logger.warning(f"duplicated indices found in {fname}")
			ret = ret[~dup_idx]
		return ret

	@classmethod
	def _load_dataset(cls, dataset: str, *,
		beta_fname: str,
		depth_fname: str = None,
		metadata_fname: str = None,
		with_repl_group: bool = False,
		cache_prefix: str = util.DEFAULT_CACHE_PREFIX
	) -> Self:
		logger.info(f"loading {dataset}")
		logger.debug(f"loading beta table")
		beta = cls._load_tsv(f"data/{dataset}/{beta_fname}", cache_prefix)
		if depth_fname is not None:
			logger.debug(f"loading depth table")
			depth = cls._load_tsv(f"data/{dataset}/{depth_fname}", cache_prefix)
			logger.debug(f"harmonizing depth and beta table indices")
			depth = depth.reindex(index=beta.index, columns=beta.columns,
				fill_value=float("nan"))
		else:
			depth = None
		# load optional data
		metadata = None
		repl_group = None
		if metadata_fname is not None:
			metadata = cls._load_tsv(f"data/{dataset}/{metadata_fname}", cache_prefix)
		if with_repl_group:
			repl_group = ReplGroup.load_dataset(dataset, cache_prefix=cache_prefix)

		return cls(beta=beta, depth=depth, metadata=metadata, repl_group=repl_group)


class BetaDepth(DatasetBase):
	@classmethod
	def load_dataset(cls, dataset: str, *,
		with_metadata: bool = False,
		with_repl_group: bool = False,
		cache_prefix: str = util.DEFAULT_CACHE_PREFIX
	) -> Self:
		new = super()._load_dataset(dataset,
			beta_fname="beta.tsv",
			depth_fname="depth.tsv",
			metadata_fname=("metadata.tsv" if with_metadata else None),
			with_repl_group=with_repl_group,
			cache_prefix=cache_prefix
		)
		return new


class BetaOnly(DatasetBase):
	@classmethod
	def load_dataset(cls, dataset: str, *,
		with_metadata: bool = False,
		with_repl_group: bool = False,
		cache_prefix: str = util.DEFAULT_CACHE_PREFIX
	) -> Self:
		new = super()._load_dataset(dataset,
			beta_fname="beta.tsv",
			metadata_fname=("metadata.tsv" if with_metadata else None),
			with_repl_group=with_repl_group,
			cache_prefix=cache_prefix
		)
		return new
