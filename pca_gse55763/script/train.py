#!/usr/bin/env python3

import argparse
import pickle

import numpy
import pandas
import sklearn
import sklearn.linear_model
import sklearn.metrics


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--beta", type=str, required=True,
		metavar="tsv",
		help="input beta value table (require)")
	ap.add_argument("-m", "--metadata", type=str, required=True,
		metavar="tsv",
		help="input metadata value table (required)")
	ap.add_argument("-a", "--alpha", type=float, required=True,
		metavar="float",
		help="alpha value (ElasticNet hyperparameter) (required)")
	ap.add_argument("-l", "--l1-ratio", type=float, required=True,
		metavar="float",
		help="lambda value (ElasticNet hyperparameter) (required)")
	ap.add_argument("-r", "--max-iter", type=int, default=100000,
		metavar="int",
		help="max. iteration in ElasticNet training [100000]")
	ap.add_argument("-o", "--output", type=str, required=True,
		metavar="pkl",
		help="trained model dumped in pickle format (required)")

	args = ap.parse_args()

	return args


def main():
	args = get_args()
	# load data
	samples, beta = load_beta(args.beta)
	age = load_age(args.metadata, samples=samples)
	# train model
	train_res = train_model(beta, age, alpha=args.alpha, l1_ratio=args.l1_ratio,
		max_iter=args.max_iter)
	# save result
	save_res(train_res, args.output)
	return


def load_beta(fname: str) -> tuple[numpy.ndarray, numpy.ndarray]:
	raw = numpy.loadtxt(fname, dtype=str, delimiter="\t", skiprows=1)
	samples = raw[:, 0]
	beta = raw[:, 1:].astype(numpy.float32)
	return samples, beta


def load_age(fname: str, samples: numpy.ndarray | list) -> numpy.ndarray:
	m = pandas.read_csv(fname, sep="\t", index_col=0)
	return numpy.asarray(m.reindex(samples)["age"].values)


def train_model(x, y, *, alpha: float, l1_ratio: float, max_iter=100000,
) -> dict:
	model = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
		max_iter=max_iter)
	model.fit(x, y)
	pred_y = model.predict(x)

	ret = {
		"alpha": alpha,
		"l1_ratio": l1_ratio,
		"model": model,
		"n_cpgs": (model.coef_ != 0).sum(),
		"mae": sklearn.metrics.mean_absolute_error(y, pred_y),
		"rmse": sklearn.metrics.root_mean_squared_error(y, pred_y),
	}
	return ret


def save_res(obj, fname: str):
	with open(fname, "wb") as fp:
		pickle.dump(obj, fp)
	return


if __name__ == "__main__":
	main()
