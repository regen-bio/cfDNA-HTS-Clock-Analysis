#!/usr/bin/env python3

import argparse

import matplotlib
import matplotlib.pyplot
import numpy
import sklearn
import sklearn.decomposition


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--beta", type=str, required=True,
		metavar="tsv",
		help="input beta value table (require)")
	ap.add_argument("-o", "--output", type=str, required=True,
		metavar="file",
		help="output pca explained variance plot (require)")

	args = ap.parse_args()

	return args


def main():
	args = get_args()
	# load data
	_, beta = load_beta(args.beta)
	# run pca
	eigens = beta_pca_eigens(beta)
	# plot
	plot_pca_eigens(eigens, args.output)
	return


def load_beta(fname: str) -> tuple[numpy.ndarray, numpy.ndarray]:
	raw = numpy.loadtxt(fname, dtype=str, delimiter="\t", skiprows=1)
	samples = raw[:, 0]
	beta = raw[:, 1:].astype(numpy.float32)
	return samples, beta


def beta_pca_eigens(beta: numpy.ndarray) -> numpy.ndarray:
	pca = sklearn.decomposition.PCA(n_components=50)
	pca.fit(beta)
	return pca.explained_variance_ratio_


def plot_pca_eigens(eigens: numpy.ndarray, fname: str):
	figure = matplotlib.pyplot.gcf()
	figure.set_size_inches(6, 4)
	axes = matplotlib.pyplot.gca()

	x = numpy.arange(len(eigens)) + 1
	cum_y = numpy.cumsum(eigens)
	axes.bar(x, eigens, width=0.9, color="#404040", zorder=2,
		label="explained variance")
	axes.step(x, cum_y, where="mid", color="#3ba1fa", zorder=3,
		label="cumulative explained variance")

	axes.set_xlim(0.5, len(eigens) + 0.5)
	axes.set_ylim(0, cum_y[-1] * 1.05)
	axes.set_xlabel("PC")
	axes.set_ylabel("explained variance")

	figure.tight_layout()
	figure.savefig(fname, dpi=300)

	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	main()
