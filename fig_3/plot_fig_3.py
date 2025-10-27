#!/usr/bin/env python3

import itertools
import json
import pdb

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot
import mpllayout
import numpy
import scipy.stats
import pandas

import pylib


def read_and_prep_simu_result(dataset: str) -> pandas.DataFrame:
	# read repl group info
	with open(f"data/{dataset}/repl.json", "r") as fp:
		repl_group: list[dict] = json.load(fp)
	repl = list(itertools.chain(*map(lambda x: x["replicates"], repl_group)))
	repl_to_subj = {r: v["subject_id"]
		for v in repl_group for r in v["replicates"]}

	# read predict results df
	# pred_df has no index
	pred_df = pandas.read_csv(f"pred_res/reprod.{dataset}.pred.tsv", sep="\t")
	# select with #cpg > 0, i.e. ignoring constant models
	pred_df = pred_df[(pred_df["n_cpgs"] > 10) & (pred_df["n_iter"] < 100000)]

	# calculate std between replicates
	pred_vals = pred_df[repl].T
	# group by subject, by applying repl_to_subj to first column
	pred_vals.index = pred_vals.index.map(repl_to_subj)
	pred_std = pred_vals.groupby(pred_vals.index).std()

	# calculate mean std across subjects for each n_cpg
	ret = pandas.DataFrame()
	ret["alpha"] = pred_df["alpha"]
	ret["log_alpha"] = numpy.log(pred_df["alpha"])
	ret["l1_ratio"] = pred_df["l1_ratio"]
	ret["n_cpgs"] = pred_df["n_cpgs"]
	ret["n_iter"] = pred_df["n_iter"]
	ret["mean_std"] = pred_std.mean().values
	ret["mae"] = pred_df["pred_mae"]
	ret["rmse"] = pred_df["pred_rmse"]

	return ret


def setup_layout(datasets: list[str], columns: list[str]) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=1.0,
		right_margin=0.2,
		top_margin=0.5,
		bottom_margin=0.7,
	)

	rsub_ncol = 2
	rsub_nrow = (len(datasets) + rsub_ncol - 1) // rsub_ncol
	rsub_w = 1.2
	rsub_h = rsub_w
	rsub_gap_w = 0.1
	rsub_gap_h = 0.1
	rsub_grid_w = rsub_w * rsub_ncol + rsub_gap_w * (rsub_ncol - 1)
	rsub_grid_h = rsub_h * rsub_nrow + rsub_gap_h * (rsub_nrow - 1)

	rsum_s = rsub_grid_w
	rsum_gap_w = 0.4
	rsum_gap_h = 0.5
	rsum_ncol = len(columns)

	alpha_full_w = (rsum_s * 3 + rsum_gap_w * 2) * 0.6
	alpha_full_h = 2.2
	alpha_full_gap_h = 0.7

	# rsub row
	for i, n in enumerate(columns):
		ref_frame = lc.get_frame(f"rsum_{n}")
		for j, ds in enumerate(datasets):
			ir = rsub_nrow - j // rsub_ncol - 1
			ic = j % rsub_ncol
			axes = lc.add_frame(f"rsub_{n}_{ds}")
			axes.set_size(rsub_w, rsub_h)
			axes.set_anchor("bottomleft", offsets=(
				(rsub_w + rsub_gap_w) * ic + (rsum_s + rsum_gap_w) * i,
				(rsub_h + rsub_gap_h) * ir,
			))

	full_alpha_offset_h = rsub_grid_h + rsum_gap_h
	full_alpha = lc.add_frame("full_alpha")
	full_alpha.set_anchor("bottomleft", offsets=(0, full_alpha_offset_h))
	full_alpha.set_size(alpha_full_w, alpha_full_h)

	layout = lc.create_figure_layout()

	# set bundles
	for n in columns:
		_list = list()
		for ds in datasets:
			_list.append(layout[f"rsub_{n}_{ds}"])
		layout[f"rsub_{n}_list"] = _list

	# massage rsub axes
	aref_axes = layout[f"rsub_{columns[0]}_{datasets[0]}"]
	for n in columns:
		cref_axes = layout[f"rsub_{n}_{datasets[0]}"]
		for idx, ds in enumerate(datasets):
			cur_axes = layout[f"rsub_{n}_{ds}"]
			# sharey for all
			cur_axes.sharey(aref_axes)
			# make logy
			cur_axes.set_yscale("log")
			# share x with axes in column
			cur_axes.sharex(cref_axes)
			# tick params
			ic = idx % rsub_ncol
			cur_axes.tick_params(which="both",
				left=(ic == 0), labelleft=(ic == 0),
				right=(ic == (rsub_ncol - 1)), labelright=False,
				bottom=True, labelbottom=(idx > len(datasets) - 3),
				top=False, labeltop=False,
			)

	layout["_rsub_nrow"] = rsub_nrow
	layout["_rsub_ncol"] = rsub_ncol
	layout["_rsub_pt_topleft"] = layout[
		f"rsub_{columns[0]}_{datasets[0]}"]
	layout["_rsub_pt_right"] = layout[
		f"rsub_{columns[-1]}_{datasets[1]}"]

	return layout


def _plot_full_alpha(axes: matplotlib.axes.Axes, *,
	datasets: list[str], dataset_cfgs: pylib.DatasetCfgLib,
	results: dict[str, pandas.DataFrame],
	shadow_topleft_axes: matplotlib.axes.Axes,
	shadow_right_axes: matplotlib.axes.Axes,
) -> list:
	axes.set_xscale("log")

	# scatter configs
	s = 40
	linewidths = 0.4

	# plot scatters backgrounds
	handles = list()
	for ds in datasets:
		df = results[ds]
		color = dataset_cfgs[ds].color
		# scatters backgrounds outline
		axes.scatter(df["alpha"], df["mae"], s=s, zorder=5,
			linewidths=linewidths * 3, edgecolors="#000000",
			facecolors="#000000",
		)
		# scatters backgrounds fill
		axes.scatter(df["alpha"], df["mae"], s=s, zorder=6,
			linewidths=linewidths, edgecolors="none",
			facecolors="#ffffff",
		)

		# scatter forgrounds
		p = axes.scatter(df["alpha"], df["mae"], s=s, zorder=10,
			linewidths=linewidths, edgecolors="none",
			facecolors=color + "c0",
			label=dataset_cfgs[ds].display_name,
		)
		handles.append(p)

	# add shadow that connects to rsub grids
	trans_inv = axes.transData.inverted()
	pt_left, pt_top = trans_inv.transform(
		shadow_topleft_axes.transAxes.transform((0, 1))
	)
	pt_right, _ = trans_inv.transform(
		shadow_right_axes.transAxes.transform((1, 1))
	)
	path = [
		(1e-5, axes.get_ylim()[1]),
		# (1e-5, 5),
		(1e-5, axes.get_ylim()[0]),
		(pt_left, pt_top),
		(pt_right, pt_top),
		(1e-2, axes.get_ylim()[0]),
		(1e-2, axes.get_ylim()[1]),
		# (1e-2, 5),
	]
	p = matplotlib.patches.Polygon(path, closed=True, clip_on=False, zorder=-1,
		edgecolor="none", facecolor="#ffb00030")
	axes.add_patch(p)

	# misc
	axes.set_xlabel(r"$\alpha$")
	axes.set_ylabel("MAE (years)", fontsize=12)

	return


def _adjust_fade(color: str, factor: float) -> str:
	hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))
	hsv[1] = max(0.0, hsv[1] * factor)
	hsv[2] = min(1.0, hsv[2] / factor)
	return matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv))


def _plot_rsub_grid(axes_list: list[matplotlib.axes.Axes], *,
	datasets: list[str], dataset_cfgs: pylib.DatasetCfgLib,
	results: dict[str, pandas.DataFrame],
	rsub_ncol: int, rsub_nrow: int,
	xkey: str, ykey: str, xlabel: str, ylabel: str,
	logx: bool = False, show_ylabel: bool = False,
) -> dict[str, tuple]:

	# plot with highlighted dataset for each
	for axes, h_ds in zip(axes_list, datasets):
		if logx:
			axes.set_xscale("log")

		df = results[h_ds]
		# plot background
		axes.scatter(df[xkey], df[ykey], s=12, linewidth=1.5, zorder=3,
			edgecolors="#000000", facecolors="#000000",
		)
		axes.scatter(df[xkey], df[ykey], s=12, zorder=4,
			edgecolors="none", facecolors="#ffffff",
		)
		# plot foreground
		axes.scatter(df[xkey], df[ykey], s=12, zorder=5,
			edgecolors="none", facecolors=dataset_cfgs[h_ds].color + "80",
			label=dataset_cfgs[h_ds].display_name,
		)

		axes.grid(color="#e0e0e0")

	# add linear regression lines
	# do this after scatter plots to ensure xlim works
	linreg_res = dict()
	for axes, ds in zip(axes_list, datasets):
		x, y = results[ds][xkey], results[ds][ykey]
		x = numpy.log10(x) if logx else x
		y = numpy.log10(y)
		_reg_res = scipy.stats.linregress(x, y)

		linreg_res[ds] = _reg_res  # add to results list

		# plot line
		x_fit = numpy.linspace(*axes.get_xlim(), 10)
		_mx = numpy.log10(x_fit) if logx else x_fit
		y_fit = numpy.pow(10, _reg_res.intercept + _reg_res.slope * _mx)
		assert numpy.isfinite(y_fit).all(), pdb.set_trace()
		line = matplotlib.lines.Line2D(x_fit, y_fit,
			linewidth=1.0, color="#000000", zorder=10,
		)
		axes.add_artist(line)  # use this to avoid updating xlim and ylim

		# add text
		t = ("slope={:.2f}\nintercept={:.1f}\nR$^2$={:.2f}\n$p$={:.1e}").format(
			_reg_res.slope,
			_reg_res.intercept,
			_reg_res.rvalue ** 2,
			_reg_res.pvalue,
		)
		if y.mean() > numpy.mean(numpy.log10(axes.get_ylim())):
			# if values clusters in the higher half of the axes, add the text in
			# the lower half of the axes
			if _reg_res.slope > 0:
				# bottom-right
				text_xy = (0.95, 0.05)
				halign = "right"
				valign = "bottom"
			else:
				# bottom-left
				text_xy = (0.05, 0.05)
				halign = "left"
				valign = "bottom"
		else:
			if _reg_res.slope > 0:
				# top-right
				text_xy = (0.95, 0.95)
				halign = "right"
				valign = "top"
			else:
				# bottom-left
				text_xy = (0.05, 0.95)
				halign = "left"
				valign = "top"
		axes.text(*text_xy, t, fontsize=8, transform=axes.transAxes, zorder=10,
			fontfamily="monospace",
			horizontalalignment=halign, verticalalignment=valign,
		)

	# add grid-wise xlabel and ylabels
	axes = axes_list[rsub_ncol * (rsub_nrow - 1)]  # always the bottomleft one
	_right, _ = axes.transAxes.inverted().transform(
		axes_list[rsub_ncol - 1].transAxes.transform((1, 0))
	)
	axes.text(_right / 2, -0.24, xlabel, clip_on=False, fontsize=12,
		transform=axes.transAxes,
		horizontalalignment="center", verticalalignment="top",
	)

	if show_ylabel:
		_, _top = axes.transAxes.inverted().transform(
			axes_list[0].transAxes.transform((0, 1))
		)
		axes.text(-0.35, _top / 2, ylabel, clip_on=False, fontsize=12,
			transform=axes.transAxes, rotation=90,
			horizontalalignment="right", verticalalignment="center",
		)

	return linreg_res


def _main():
	datasets = ["RB_MSA", "RB_935K", "RB_GDNA_GALAXY", "RB_GDNA_TWIST",
		"RB_GALAXY", "RB_TWIST"]
	columns = ["ncpg", "alpha", "lambda"]

	full_simu_res: dict = {ds: read_and_prep_simu_result(ds) for ds in datasets}
	# filt_simu_res = {k: v[(v["alpha"] < 1e-2) & (v["mae"] < 5)]
	# for k, v in full_simu_res.items()}
	filt_simu_res = {k: v[v["alpha"] < 1e-2] for k, v in full_simu_res.items()}

	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	layout = setup_layout(datasets=datasets, columns=columns)
	figure = layout["figure"]

	# plot alpha all-range results
	handles = _plot_full_alpha(layout["full_alpha"],
		datasets=datasets, dataset_cfgs=dataset_cfgs, results=full_simu_res,
		shadow_topleft_axes=layout["_rsub_pt_topleft"],
		shadow_right_axes=layout["_rsub_pt_right"]
	)
	# legend
	legend = layout["full_alpha"].legend(handles=handles,
		loc=3, bbox_to_anchor=(1.07, -0.02), ncol=2,
		frameon=True, handlelength=0.75, title="Datasets (through A-G)",
	)

	# plot rsub grids
	# record regression results
	linreg_res = dict()
	_res = _plot_rsub_grid(layout["rsub_ncpg_list"],
		datasets=datasets, dataset_cfgs=dataset_cfgs, results=filt_simu_res,
		xkey="n_cpgs", ykey="mean_std",
		rsub_ncol=layout["_rsub_ncol"], rsub_nrow=layout["_rsub_nrow"],
		xlabel="Number of CpGs", ylabel="RD (years)",
		logx=True, show_ylabel=True,
	)
	linreg_res["ncpg"] = _res

	_res = _plot_rsub_grid(layout["rsub_alpha_list"],
		datasets=datasets, dataset_cfgs=dataset_cfgs, results=filt_simu_res,
		xkey="alpha", ykey="mean_std",
		rsub_ncol=layout["_rsub_ncol"], rsub_nrow=layout["_rsub_nrow"],
		xlabel="$\\alpha$", ylabel="Replicate Disagreement (years)",
		logx=True, show_ylabel=False,
	)
	linreg_res["alpha"] = _res

	_res = _plot_rsub_grid(layout["rsub_lambda_list"],
		datasets=datasets, dataset_cfgs=dataset_cfgs, results=filt_simu_res,
		xkey="l1_ratio", ykey="mean_std",
		rsub_ncol=layout["_rsub_ncol"], rsub_nrow=layout["_rsub_nrow"],
		xlabel="$\\lambda$", ylabel="Replicate Disagreement (years)",
		logx=False, show_ylabel=False,
	)
	linreg_res["lambda"] = _res

	figure.savefig("fig_3.svg", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
