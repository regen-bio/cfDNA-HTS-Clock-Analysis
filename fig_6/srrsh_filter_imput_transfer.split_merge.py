#!/usr/bin/env python3

import glob
import os
import pickle

import pylib


def _main():
	merged = dict()

	for fname in glob.glob(os.path.join(".tmp", "srrsh_filter_imput_transfer.age_pred.*.pkl")):
		with open(fname, "rb") as fp:
			obj = pickle.load(fp)
		pylib.util.merge_dict_inplace(merged, obj, overwrite=True)

	# save merged results
	with open("srrsh_filter_imput_transfer.age_pred.pkl", "wb") as fp:
		pickle.dump(merged, fp)

	return


if __name__ == "__main__":
	_main()
