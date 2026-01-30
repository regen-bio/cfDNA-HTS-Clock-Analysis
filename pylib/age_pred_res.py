#!/usr/bin/env python3

import dataclasses
import functools
import os
import pdb
from typing import NoReturn, Self, Sequence

import numpy
import pandas

from . import logger, util, dataset


@dataclasses.dataclass
class AgePredRes(object):
	data: pandas.DataFrame

	@classmethod
	def from_txt(cls, fname: str, *,
		cache_prefix: str = util.DEFAULT_CACHE_PREFIX
	) -> Self:
		fname = util.try_load_cache(fname, cache_prefix)
		data = pandas.read_csv(fname, sep="\t", index_col=0)
		return cls(data=data)

	def _calc_stat_along_column(self, repl_group: dataset.ReplGroup | None,
		data_columns: Sequence[str], by_subject: bool = False,
	) -> pandas.DataFrame:
		# lazy load
		from sklearn.metrics import mean_absolute_error, root_mean_squared_error

		true_age = self.data["age"]
		# pred_age = self.reindex(columns=data_columns)
		pred_age = self.data.reindex(columns=data_columns)

		# calculate mae and rmse
		if by_subject:
			ac_true_age = true_age.groupby(repl_group.repl_to_subj).mean()
			ac_pred_age = pred_age.groupby(repl_group.repl_to_subj).mean()
		else:
			ac_true_age = true_age
			ac_pred_age = pred_age

		mae = [mean_absolute_error(ac_true_age, v)
			for v in ac_pred_age.T.values]
		rmse = [root_mean_squared_error(ac_true_age, v)
			for v in ac_pred_age.T.values]

		if repl_group is not None:
			# calculate repl_mad and repl_std
			# by_subject don't have an effect on these
			repl_mad = list()
			repl_std = list()

			for c in pred_age.columns:
				# calculate group mean and expand to original index
				g_mean = pred_age[c].groupby(repl_group.repl_to_subj).mean()
				# calculate residual between pred and group mean
				resid = pandas.Series(numpy.nan, index=pred_age.index)
				for idx in resid.index:
					resid[idx] = pred_age.loc[idx, c] - g_mean[repl_group.repl_to_subj[idx]]
				repl_mad.append(resid.abs().groupby(repl_group.repl_to_subj).mean().mean())
				repl_std.append(resid.groupby(repl_group.repl_to_subj).std(ddof=1).mean())
		else:
			repl_mad = [float("nan")] * len(pred_age.columns)
			repl_std = [float("nan")] * len(pred_age.columns)

		# return data as a dataframe
		ret = pandas.DataFrame(index=pred_age.columns)
		ret["mae"] = mae
		ret["rmse"] = rmse
		ret["repl_mad"] = repl_mad
		ret["repl_std"] = repl_std

		return ret

	def calc_stat(self, data_columns: Sequence[str], *,
		repl_group: dataset.ReplGroup = None, along_row: bool = False,
		by_subject: bool = False,
	) -> pandas.DataFrame:
		if along_row:
			raise NotImplementedError
		else:
			ret = self._calc_stat_along_column(repl_group, data_columns,
				by_subject=by_subject)
		return ret

	def calc_delta_age(self, data_columns: Sequence[str]) -> pandas.DataFrame:
		true_age = self.data["age"]
		pred_age = self.data.reindex(columns=data_columns).copy()
		delta_age = pred_age - true_age[:, numpy.newaxis]
		return delta_age
