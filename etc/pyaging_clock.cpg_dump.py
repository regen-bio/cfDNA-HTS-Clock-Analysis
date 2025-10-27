#!/usr/bin/env python3

import json
import os

import torch
import tqdm


if __name__ == "__main__":
	pyaging_data = "../pyaging_data"

	# load metadata
	metadata = torch.load(os.path.join(pyaging_data, "all_clock_metadata.py"))

	clock_cpgs = dict()
	for clock_name in tqdm.tqdm(metadata.keys()):
		clock_file = os.path.join(pyaging_data, f"{clock_name}.pt")
		clock_model = torch.load(clock_file, weights_only=False)

		clock_cpgs[clock_name] = clock_model.features

	# save data
	with open("pyaging_clock_cpgs.json", "w") as fp:
		json.dump(clock_cpgs, fp, indent="\t", sort_keys=True)
