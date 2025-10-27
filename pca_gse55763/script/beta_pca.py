#!/usr/bin/env python3

import argparse
import pickle

import numpy
import sklearn
import sklearn.decomposition


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--beta", type=str, required=True,
		metavar="tsv",
		help="input beta value table (require)")
	ap.add_argument("-n", "--n-components", type=int, required=True,
		metavar="int",
		help="number of PCA components to compute (required)")
	ap.add_argument("-o", "--output-beta", type=str, required=True,
		metavar="tsv",
		help="output PCA-transformed beta value table (require)")
	ap.add_argument("-p", "--output-pca", type=str, required=True,
		metavar="pkl",
		help="output PCA model (require)")

	args = ap.parse_args()

	return args


def main():
	args = get_args()
	# load data
	samples, beta = load_beta(args.beta)
	# run pca
	pca, beta_tr = beta_pca(beta, args.n_components)
	# save results
	save_pca_model(args.output_pca, pca)
	save_transformed_beta(args.output_beta, samples, beta_tr)
	return


def load_beta(fname: str) -> tuple[numpy.ndarray, numpy.ndarray]:
	raw = numpy.loadtxt(fname, dtype=str, delimiter="\t", skiprows=1)
	samples = raw[:, 0]
	beta = raw[:, 1:].astype(numpy.float32)
	return samples, beta


def beta_pca(beta: numpy.ndarray, n_components: int,
) -> tuple[sklearn.decomposition.PCA, numpy.ndarray]:
	pca = sklearn.decomposition.PCA(n_components=n_components)
	return pca, pca.fit_transform(beta)


def save_pca_model(fname: str, pca: sklearn.decomposition.PCA):
	with open(fname, "wb") as fp:
		pickle.dump(pca, fp)
	return


def save_transformed_beta(fname: str, samples: list | numpy.ndarray,
	data: numpy.ndarray
):
	with open(fname, "w") as fp:
		# header line
		print("\t" + ("\t").join(map(lambda x: f"pc_{x + 1}",
			range(data.shape[1]))), file=fp)
		# data lines
		for s, d in zip(samples, data):
			print(s + "\t" + ("\t").join(map(lambda x: "%.6f" % x, d)), file=fp)
	return


if __name__ == "__main__":
	main()
