#!/usr/bin/env python3

import pandas

with open("input/samples.list", "r") as fp:
	samples = fp.read().splitlines()

with open("input/common_cpgs.list", "r") as fp:
	cpgs = fp.read().splitlines()

raw = pandas.read_csv("input/beta.tsv", sep="\t", index_col=0)

flt = raw.loc[cpgs, samples].T

flt.to_csv("output/shrink_table.tsv", sep="\t", float_format="%.4f", index=True)
