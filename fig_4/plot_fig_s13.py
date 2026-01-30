#!/usr/bin/env python3

import json
import pdb

import matplotlib
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot
import mpllayout
import numpy
import threadpoolctl

import pylib


class DatasetPlotCfg(dict):
	@classmethod
	def from_file(cls, path: str):
		with open(path, "r") as fp:
			raw = json.load(fp)
		if not isinstance(raw, dict):
			raise ValueError(f"invalid config file: {path}")
		return cls(raw)

	def __get__(self, key: str) -> dict:
		ret = super().__getitem__(key)
		if ret.get("ref", None) is not None:
			ret = self[key["ref"]]  # return the cross-referenced config
		return ret


def _esti_depth_pdf_hist(vals: numpy.ndarray, max_val: float = None,
	normalize: float = 1.0,
) -> tuple[numpy.ndarray, numpy.ndarray]:
	if max_val is None:
		max_val = numpy.nanmax(vals)
	bins = numpy.arange(max_val + 2) - 0.5
	hist = numpy.histogram(vals, bins=bins, density=True)[0] * normalize
	return hist


def _esti_depth_pdfs(data: dict[str, pylib.dataset.FinalFullBetaDepth],
	max_depth_quantile: float = 0.99,
) -> dict[str]:
	# determine the max depth
	all_depths = numpy.hstack([v.depth.values.flatten() for v in data.values()])
	n_total = len(all_depths)
	max_depth = int(numpy.quantile(all_depths, max_depth_quantile))
	pos = numpy.arange(max_depth + 1)

	ret = {
		"pos": pos,
		"pdf": dict(),
	}
	for ds, d in data.items():
		ill_mask: numpy.ndarray = (d.beta.values) == 0 | (d.beta.values == 1)
		norm_mask: numpy.ndarray = (~ill_mask) & numpy.isfinite(d.beta.values)

		pdfs = dict()
		for k, mask in zip(["ill", "normal"], [ill_mask, norm_mask]):
			pdfs[k] = _esti_depth_pdf_hist(d.depth[mask].values.flatten(),
				max_val=max_depth,
				normalize=mask.sum() / n_total,
			)

		ret["pdf"][ds] = pdfs
	ret["pos"] = pos

	return ret


def setup_layout(n_datasets: int) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=1.5,
		right_margin=0.2,
		top_margin=0.5,
		bottom_margin=0.7,
	)

	axes = lc.add_frame("axes")
	axes.set_anchor("bottomleft")
	axes.set_size(4, 0.6 * n_datasets)

	layout = lc.create_figure_layout()

	return layout


def _plot_rep_beta_vs_depth(axes: matplotlib.axes.Axes, *,
	datasets: list[str],
	pdf_dict: dict[str],
	dataset_cfgs: pylib.DatasetCfgLib,
	scale: float = 1,
):
	pos = pdf_dict["pos"]

	for i, ds in enumerate(datasets):
		pdf = pdf_dict["pdf"][ds]

		# normal cpgs
		axes.plot(pos, pdf["normal"] * scale + i,
			linewidth=0.5, color="#808080", zorder=10,
		)
		axes.fill_between(pos, i, pdf["normal"] * scale + i,
			edgecolor="none", facecolor="#80808080", zorder=5
		)
		# ill cpgs
		ill_color = dataset_cfgs[ds].color
		axes.plot(pos, pdf["ill"] * scale + i,
			linewidth=0.5, color="#000000", zorder=11,
		)
		axes.fill_between(pos, i, pdf["ill"] * scale + i,
			edgecolor="none", facecolor=ill_color + "80", zorder=6
		)

	# add legend
	handles = []
	for ds in datasets:
		cfg = dataset_cfgs[ds]
		p = matplotlib.patches.Patch(linewidth=0.5, edgecolor="#000000",
			facecolor=cfg.color + "80",
			label="0/1 betas in %s" % cfg.display_name)
		handles.append(p)
	# add 'others'
	p = matplotlib.patches.Patch(linewidth=0.5, edgecolor="#808080",
		facecolor="#80808080", label="Other beta values")
	handles.append(p)
	legend = axes.legend(handles=handles, loc=1, bbox_to_anchor=[0.98, 0.98],
		frameon=False, handlelength=0.75, fontsize=8,
	)

	# misc
	axes.tick_params(
		left=True, labelleft=True,
		right=False, labelright=False,
		bottom=True, labelbottom=True,
		top=False, labeltop=False,
	)
	axes.grid(linewidth=0.5, color="#d0d0d0")
	axes.set_xlabel("Depth")
	axes.set_yticks(numpy.arange(len(datasets)))
	axes.set_yticklabels([dataset_cfgs[ds].display_name for ds in datasets])
	axes.set_title("PDF of 0/1 beta and other beta values")

	return


def _main():
	# datasets is order-related
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	n_datasets = len(datasets)

	# loading data
	data = {ds: pylib.dataset.FinalFullBetaDepth.load_dataset(ds)
		for ds in datasets}

	# dataset appearance configs
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# calculate pdfs of ill and non-ill pdfs
	pdf_dict = _esti_depth_pdfs(data, max_depth_quantile=0.99)

	layout = setup_layout(n_datasets)
	figure = layout["figure"]

	_plot_rep_beta_vs_depth(layout["axes"],
		datasets=datasets,
		pdf_dict=pdf_dict,
		dataset_cfgs=dataset_cfgs,
		scale=100,
	)

	figure.savefig("fig_s13.svg", dpi=600)
	figure.savefig("fig_s13.png", dpi=600)
	figure.savefig("fig_s13.pdf", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	with threadpoolctl.threadpool_limits(limits=24, user_api="blas"):
		_main()
