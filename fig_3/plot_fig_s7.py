#!/usr/bin/env python3

import os
import pdb

import matplotlib
import matplotlib.pyplot
import matplotlib.axes
import matplotlib.lines
import mpllayout
import numpy
import scipy.stats
import pandas

import pylib


def setup_layout(n_axes) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.8,
		right_margin=2.0,
		top_margin=0.2,
		bottom_margin=0.7,
	)

	nrows = int(numpy.ceil(numpy.sqrt(n_axes)))
	ncols = int(numpy.ceil(n_axes / nrows))

	axes_w = 2
	axes_h = axes_w
	pad_w = 0.4
	pad_h = 0.3

	for i in range(n_axes):
		ri = nrows - i // ncols - 1  # flip rows
		ci = i % ncols

		axes = lc.add_frame(f"axes_{i}")
		axes.set_size(axes_w, axes_h)
		axes.set_anchor("bottomleft",
			offsets=((axes_w + pad_w) * ci, (axes_h + pad_h) * ri))

	layout = lc.create_figure_layout()
	layout["nrows"] = nrows
	layout["ncols"] = ncols

	return layout


def prep_age_pred_std(dataset: str, *, clock_cpgs: dict[list],
	exclude_clocks: set[str] | None = None,
) -> tuple[list[str], pandas.DataFrame]:
	# load repl group data
	repl_group = pylib.dataset.ReplGroup.load_dataset(dataset)

	# load age prediction data
	fname = f"data/{dataset}/age_pred.tsv"
	age_pred = pylib.age_pred_res.AgePredRes.from_txt(fname)

	# extract error propagation data for each clock
	clocks = [c for c in age_pred.data.columns if c in clock_cpgs]
	if exclude_clocks is not None:
		clocks = [c for c in clocks if c not in exclude_clocks]

	# filter age_pred to only include clocks
	age_pred_stat = age_pred.calc_stat(clocks, repl_group=repl_group)

	return clocks, age_pred_stat["repl_mad"]


def _plot_age_pred_vs_clock_coef(axes: matplotlib.axes.Axes, dataset: str, *,
	dataset_cfg: pylib.DatasetCfg,
	clock_cpgs: dict[list],
	clock_cfgs: pylib.ClockCfgLib,
	show_xlabel: bool = False, show_ylabel: bool = False,
	exclude_clocks: set[str] = None,
) -> tuple[list[str], tuple]:
	clocks, age_pred_rd = prep_age_pred_std(dataset, clock_cpgs=clock_cpgs,
		exclude_clocks=exclude_clocks)

	# plot age pred std vs # clock coefs
	x = [len(clock_cpgs[c]) for c in clocks]
	y = age_pred_rd.values
	# set log scale
	axes.set_xscale("log")
	axes.set_yscale("log")
	for i, c in enumerate(clocks):
		cfg = clock_cfgs[c]
		axes.scatter([x[i]], [y[i]], marker=cfg.marker, s=60, zorder=5,
			linewidths=0.5, edgecolors="#000000",
			facecolors=cfg.color + "80",
		)
	# add regression line
	linreg = scipy.stats.linregress(numpy.log(x), numpy.log(y))
	fit_x = axes.get_xlim()
	fit_y = numpy.exp(linreg.intercept) * numpy.array(fit_x) ** linreg.slope
	line = matplotlib.lines.Line2D(fit_x, fit_y, color="#808080", linestyle="-",
		linewidth=2.0, zorder=3)
	axes.add_artist(line)
	# add regression equation
	s = f"ln(y)={linreg.slope:.2f}ln(x)+{linreg.intercept:.2f}\n" \
		+ f"R={linreg.rvalue:.2f}\n$p$={linreg.pvalue:.1e}"
	axes.text(0.05, 0.05, s,
		transform=axes.transAxes, fontsize=10, zorder=5,
		horizontalalignment="left", verticalalignment="bottom")

	# add plot label
	axes.text(0.95, 0.95, dataset_cfg.display_name, fontsize=12, fontweight="bold",
		transform=axes.transAxes, color=dataset_cfg.color, zorder=5,
		horizontalalignment="right", verticalalignment="top")

	# misc
	axes.grid(linewidth=0.5, color="#c0c0c0", zorder=2)

	if show_xlabel:
		axes.set_xlabel("#CpGs in clock", fontsize=12)
	if show_ylabel:
		axes.set_ylabel("RD (years)", fontsize=12)

	return clocks, linreg


def _main():
	# order-related
	datasets = ["RB_MSA", "RB_935K", "RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY",
		"RB_TWIST", "GSE83944", "GSE247193", "GSE247195", "GSE247197", "GSE55763"]
	exclude_clocks: str[str] = {"altumage"}

	# load clock cpg data
	clock_cpgs = pylib.clock.load_clock_features()

	clock_cfgs = pylib.clock_cfg.ClockCfgLib.from_json()

	n_datasets = len(datasets)
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	layout = setup_layout(n_datasets)
	figure = layout["figure"]
	nrows = layout["nrows"]
	ncols = layout["ncols"]

	# share x and y axes
	for i in range(1, n_datasets):
		layout[f"axes_{i}"].sharex(layout["axes_0"])
		layout[f"axes_{i}"].sharey(layout["axes_0"])

	all_clocks = set()
	reg_res = dict[str, scipy.stats.linregress]()
	for i, dataset in enumerate(datasets):
		_clocks, _linreg = _plot_age_pred_vs_clock_coef(
			layout[f"axes_{i}"], dataset,
			dataset_cfg=dataset_cfgs[dataset],
			clock_cpgs=clock_cpgs,
			clock_cfgs=clock_cfgs,
			show_xlabel=(n_datasets - i <= ncols),
			show_ylabel=(i % ncols == 0),
			exclude_clocks=exclude_clocks,
		)
		all_clocks.update(_clocks)
		reg_res[dataset] = _linreg

	# legend
	handles = list()
	axes = layout["axes_0"]
	for c in sorted(all_clocks):
		cfg = clock_cfgs[c]
		p = axes.scatter([], [], marker=cfg.marker, s=60, zorder=5, linewidths=0.5,
			edgecolors="#000000", facecolors=cfg.color + "80", label=cfg.key,
		)
		handles.append(p)
	figure.legend(handles=handles, loc=1, bbox_to_anchor=(0.98, 0.98),
		handlelength=0.75, fontsize=10, frameon=False,
		title="Linear Regression\nClocks", title_fontsize=12)

	figure.savefig("fig_s7.svg", dpi=600)
	matplotlib.pyplot.close(figure)

	# save regression results as table
	cols = ["slope", "intercept", "rvalue", "pvalue"]
	reg_res_df = pandas.DataFrame(
		index=(dataset_cfgs[d].display_name for d in datasets),
		columns=cols,
	)
	for c in cols:
		reg_res_df[c] = [getattr(reg_res[d], c) for d in datasets]
	reg_res_df.to_csv("table_s3.tsv", index=True,
		sep="\t")

	return


if __name__ == "__main__":
	_main()
