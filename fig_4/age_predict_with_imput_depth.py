#!/usr/bin/env python3

import argparse
import itertools
import pickle
import sys

import numpy
import pandas
import pyaging
import tqdm


class CommaSepIntList(list):
	@classmethod
	def from_str(cls, s: str):
		return cls(s.split(","))


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--beta-table", type=str, required=True,
		metavar="tsv",
		help="table of beta values (required)")
	ap.add_argument("-d", "--depth-table", type=str, required=True,
		metavar="tsv",
		help="table of depth values (required)")
	ap.add_argument("-m", "--metadata", type=str, required=True,
		metavar="tsv",
		help="table of metadata (required)")
	ap.add_argument("-C", "--clocks", type=CommaSepIntList.from_str,
		required=True, metavar="clock[,clock,...]",
		help="comma-separated list of clocks for age prediction (required)")
	ap.add_argument("-I", "--imputations", type=CommaSepIntList.from_str,
		default=["mean", "median", "constant", "knn"],
		metavar="str[,str[,...]",
		help="comma-seprarted list of imputation methods "
			"['mean', 'median', 'constant', 'knn']")
	ap.add_argument("-D", "--max-depths", type=CommaSepIntList.from_str,
		default=[0], metavar="int[,int[,...]",
		help="comma-separated integer depth values to ignore 0/1 beta sites if "
			"its depth is larger than this value [0]; 0/negative values mean no"
			" such depth restriction during filtering")
	ap.add_argument("-o", "--output", type=str, required=True,
		metavar="pkl",
		help="output results bundled as a pickled dict (required)")

	# parse and refine args
	args = ap.parse_args()
	args.max_depths = [int(v) for v in args.max_depths]
	return args


def _filter_beta(beta: pandas.DataFrame, depth: pandas.DataFrame,
	bad_values: tuple[float], max_depth: int,
) -> pandas.DataFrame:
	mask = None
	for v in bad_values:
		if mask is None:
			mask = (beta == v)
		else:
			mask |= (beta == v)

	if max_depth > 0:
		mask = mask & (depth <= max_depth)

	# return a copied version of beta
	ret = beta.copy()
	ret[mask] = numpy.nan  # fill them as nan
	return ret


def _predict_age_single(beta: pandas.DataFrame, depth: pandas.DataFrame,
	meta: pandas.DataFrame, *, bad_values: tuple[float], imput: str,
	clocks: list[str], max_depth: int = 0,
):
	pbeta = _filter_beta(beta, depth, bad_values, max_depth)
	pbeta = pbeta.reindex(meta.index)
	pbeta = pandas.concat([meta, pbeta], axis=1)
	pbeta = pyaging.pp.epicv2_probe_aggregation(pbeta)
	adata = pyaging.pp.df_to_adata(pbeta,
		metadata_cols=meta.columns,
		imputer_strategy=imput,
	)
	pyaging.pred.predict_age(adata, clocks, dir="../pyaging_data")
	return adata


def predict_age(beta: pandas.DataFrame, depth: pandas.DataFrame,
	meta: pandas.DataFrame, *, clocks: list[str], imput_list: list[str],
	max_depths: list[int], bad_values: tuple[float] = (0.0, 1.0),
) -> dict[tuple[str, int], pandas.DataFrame]:
	# print summarized info
	print("summary: ", file=sys.stderr)
	print(f"imputation methods: {imput_list}", file=sys.stderr)
	print(f"max depths: {max_depths}", file=sys.stderr)
	print(f"clocks: {clocks}", file=sys.stderr)

	ret = dict()
	for imput, max_depth in tqdm.tqdm(itertools.product(imput_list, max_depths),
		total=len(imput_list) * len(max_depths)
	):
		# predicting
		print(f"predicting age: {imput} at depth {max_depth}", file=sys.stderr)
		adata = _predict_age_single(beta, depth, meta,
			bad_values=bad_values, imput=imput, clocks=clocks,
			max_depth=max_depth,
		)
		ret[(imput, max_depth)] = adata.obs.copy()
	return ret


def main():
	args = get_args()

	# load beta
	print(f"loading beta table: {args.beta_table}", file=sys.stderr)
	beta = pandas.read_csv(args.beta_table, sep="\t", index_col=0).T

	# load depth and align to beta
	print(f"loading depth table: {args.depth_table}", file=sys.stderr)
	depth = pandas.read_csv(args.depth_table, sep="\t", index_col=0).T
	depth = depth.reindex(index=beta.index, columns=beta.columns)

	# load meta
	print(f"loading metadata table: {args.metadata}", file=sys.stderr)
	meta = pandas.read_csv(args.metadata, sep="\t", index_col=0)

	# predicting and save
	print("predicting age:", file=sys.stderr)
	age_pred = predict_age(beta, depth, meta,
		clocks=args.clocks,
		imput_list=args.imputations,
		max_depths=args.max_depths,
	)

	# saving results
	print(f"saving results: {args.output}", file=sys.stderr)
	with open(args.output, "wb") as fp:
		pickle.dump(age_pred, fp)
	return


if __name__ == "__main__":
	main()
