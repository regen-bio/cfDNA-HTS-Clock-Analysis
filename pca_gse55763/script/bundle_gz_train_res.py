#!/usr/bin/env python3

import argparse
import gzip
import os
import pickle


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input-dir", type=str, required=True,
		metavar="dir",
		help="input training result directory (required)")
	ap.add_argument("-o", "--output", type=str, required=True,
		metavar="pkl.gz",
		help="output bundled training results (required)")

	args = ap.parse_args()

	return args


def main():
	args = get_args()
	# load training results
	bundle: list[dict] = []
	for f in os.scandir(args.input_dir):
		if (not f.is_file()) or (not f.name.endswith(".pkl")):
			continue
		with open(f, "rb") as f:
			bundle.append(pickle.load(f))
	# save bundled results
	with gzip.open(args.output, "w") as fp:
		pickle.dump(bundle, fp)
	return


if __name__ == "__main__":
	main()
