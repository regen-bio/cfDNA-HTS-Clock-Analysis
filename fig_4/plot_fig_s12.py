#!/usr/bin/env python3

import pdb
import pickle
import sys

import matplotlib
import matplotlib.axes
import matplotlib.pyplot
import mpllayout
import numpy
import threadpoolctl

import pylib


def _prep_heatmap_data(data: dict[str, pylib.dataset.RawBetaDepth]) -> dict[str]:
	# determine:
	#   beta bin
	#   depth bin
	#   histograms
	#   histogram vmax

	# all_beta = numpy.hstack([v.beta.values.flatten() for v in data.values()])
	beta_bin = numpy.linspace(0, 1, 98)  # use 98 since 97 is a prime

	all_depth = numpy.hstack([v.depth.values.flatten() for v in data.values()])
	depth_bin_max = int(numpy.quantile(all_depth, 0.99))
	depth_bin = numpy.linspace(0, depth_bin_max, depth_bin_max + 1)

	hist_2d = dict[str, numpy.ndarray]()
	for ds, v in data.items():
		_hist, *_ = numpy.histogram2d(
			v.beta.values.flatten(),
			v.depth.values.flatten(),
			bins=(beta_bin, depth_bin),
			density=False,
		)
		hist_2d[ds] = _hist.astype(int)

	# save as pickle
	fname = "output/beta_vs_depth_hist_2d.pkl"
	print(f"saving histogram matrix: {fname}", file=sys.stderr)
	export = {
		"beta_bin": beta_bin,
		"depth_bin": depth_bin,
		"hist_2d": hist_2d,
	}
	with open(fname, "wb") as fp:
		pickle.dump(export, fp)

	# normalize hist_2d
	for ds, v in hist_2d.items():
		hist_2d[ds] = v / v.sum()
	all_hist_vals = numpy.hstack(list(hist_2d.values())).flatten()
	hist_vmax = numpy.quantile(all_hist_vals, 0.99)

	ret = {
		"beta_bin": beta_bin,
		"depth_bin": depth_bin,
		"hist_2d": hist_2d,
		"hist_vmax": hist_vmax,
	}

	return ret


def setup_layout(n_datasets: int, ncols: int = 2) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.7,
		right_margin=0.2,
		top_margin=0.5,
		bottom_margin=0.7,
	)

	heatmap_s = 3.0
	heatmap_gap = 0.1
	hist_d = heatmap_s / 4
	block_s = heatmap_s + heatmap_gap + hist_d
	block_gap = 0.6
	cbar_gap_w = 0.3
	cbar_w = 0.1

	nrows = (n_datasets + ncols - 1) // ncols

	for i in range(n_datasets):
		ri = nrows - i // ncols - 1
		ci = i % ncols

		heatmap = lc.add_frame(f"heatmap_{i}")
		heatmap.set_anchor("bottomleft", offsets=(
			(block_s + block_gap) * ci,
			(block_s + block_gap) * ri,
		))
		heatmap.set_size(heatmap_s, heatmap_s)

		beta = lc.add_frame(f"beta_{i}")
		beta.set_anchor("bottomleft", heatmap, "topleft",
			offsets=(0, heatmap_gap))
		beta.set_size(heatmap_s, hist_d)

		depth = lc.add_frame(f"depth_{i}")
		depth.set_anchor("bottomleft", heatmap, "bottomright",
		    offsets=(heatmap_gap, 0))
		depth.set_size(hist_d, heatmap_s)

	layout = lc.create_figure_layout()
	layout["nrows"] = nrows
	layout["ncols"] = ncols

	return layout


def _plot_rep_beta_vs_depth(*,
	heatmap_axes: matplotlib.axes.Axes,
	beta_hist_axes: matplotlib.axes.Axes,
	depth_hist_axes: matplotlib.axes.Axes,
	hist_2d: numpy.ndarray,
	beta_bin: numpy.ndarray,
	depth_bin: numpy.ndarray,
	heatmap_vmax: float = None,
	dataset_cfg: pylib.DatasetCfg,
):
	# plot heatmap
	p = heatmap_axes.pcolor(beta_bin, depth_bin, hist_2d.T, cmap="viridis",
		vmin=0, vmax=heatmap_vmax, rasterized=True,
	)
	# add dataset name as axes label and
	s = dataset_cfg.display_name
	heatmap_axes.text(0.05, 0.95, s,
		fontsize=12, color="#ffffff", zorder=5, transform=heatmap_axes.transAxes,
		horizontalalignment="left", verticalalignment="top",
	)
	# misc
	heatmap_axes.set_xlabel("Beta")
	heatmap_axes.set_ylabel("Depth")

	# plot beta curve
	beta_hist_y = hist_2d.sum(axis=1)
	beta_hist_x = (beta_bin[:-1] + beta_bin[1:]) / 2
	beta_hist_axes.plot(beta_hist_x, beta_hist_y, linewidth=0.5,
		color="#000000", zorder=6,
	)
	beta_hist_axes.fill_between(beta_hist_x, 0, beta_hist_y, zorder=5,
		edgecolor="none", facecolor=dataset_cfg.color + "40",
	)
	beta_hist_axes.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	for sp in beta_hist_axes.spines.values():
		sp.set_visible(False)
	beta_hist_axes.set_xlim(*heatmap_axes.get_xlim())
	beta_hist_axes.set_ylim(0, None)

	# plot depth curve
	depth_hist_x = hist_2d.sum(axis=0)
	depth_hist_y = (depth_bin[:-1] + depth_bin[1:]) / 2
	depth_hist_axes.plot(depth_hist_x, depth_hist_y, linewidth=0.5,
		color="#000000", zorder=6)
	depth_hist_axes.fill_betweenx(depth_hist_y, 0, depth_hist_x, zorder=5,
		edgecolor="none", facecolor=dataset_cfg.color + "40",
	)
	depth_hist_axes.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	for sp in depth_hist_axes.spines.values():
		sp.set_visible(False)
	depth_hist_axes.set_xlim(0, None)
	depth_hist_axes.set_ylim(*heatmap_axes.get_ylim())

	return


def _main():
	# DATASETS is order-related
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	n_datasets = len(datasets)

	# loading data
	data = {ds: pylib.dataset.RawBetaDepth.load_dataset(ds) for ds in datasets}

	# dataset appearance configs
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# calculate stats for heatmap plotting
	heatmap_data = _prep_heatmap_data(data)

	layout = setup_layout(n_datasets=n_datasets, ncols=2)
	figure = layout["figure"]

	for i, ds in enumerate(datasets):
		p = _plot_rep_beta_vs_depth(
			heatmap_axes=layout[f"heatmap_{i}"],
			beta_hist_axes=layout[f"beta_{i}"],
			depth_hist_axes=layout[f"depth_{i}"],
			hist_2d=heatmap_data["hist_2d"][ds],
			beta_bin=heatmap_data["beta_bin"],
			depth_bin=heatmap_data["depth_bin"],
			heatmap_vmax=heatmap_data["hist_vmax"],
			dataset_cfg=dataset_cfgs[ds],
		)

	figure.savefig("fig_s12.svg", dpi=600)
	figure.savefig("fig_s12.png", dpi=600)
	figure.savefig("fig_s12.pdf", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	with threadpoolctl.threadpool_limits(limits=24, user_api="blas"):
		_main()
