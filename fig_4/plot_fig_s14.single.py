#!/usr/bin/env python3

import collections
import json
import matplotlib
import matplotlib.pyplot
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import mpllayout
import numpy
import os
import pandas

import pylib


def setup_layout(nrows: int, ncols: int, *, dpi: int = 300) -> int:
	lc = mpllayout.LayoutCreator(
		left_margin=1.2,
		right_margin=0.5,
		top_margin=0.5,
		bottom_margin=0.7,
	)

	grid_w = ncols * 0.2
	grid_h = nrows * 0.2

	grid = lc.add_frame("grid")
	grid.set_size(grid_w, grid_h)
	grid.set_anchor("bottomleft")

	hist_top = lc.add_frame("hist_top")
	hist_top.set_anchor("bottomleft", grid, "topleft", offsets=(0, 0.05))
	hist_top.set_size(grid_w, 1.2)

	hist_right = lc.add_frame("hist_right")
	hist_right.set_anchor("bottomleft", grid, "bottomright", offsets=(0.1, 0))
	hist_right.set_size(2.0, grid_h)

	layout = lc.create_figure_layout()
	figure = layout["figure"]
	figure.set(dpi=dpi)
	return layout


def _hbar_step(
	axes: matplotlib.axes.Axes,
	x: numpy.ndarray,
	y: numpy.ndarray,
	**kw,
) -> matplotlib.lines.Line2D:
	assert len(y) == len(x) + 1
	path_x, path_y = list(), list()
	for i in range(len(x)):
		path_x.extend([x[i], x[i]])
		path_y.extend([y[i], y[i + 1]])

	line = matplotlib.lines.Line2D(path_x, path_y, **kw)
	axes.add_line(line)
	return line


def _suffix_formatter_func(v: float, pos: int) -> str:
	if v >= 1e6:
		ret = f"{round(v / 1e6)}M"
	elif v >= 1e3:
		ret = f"{round(v / 1e3)}K"
	else:
		ret = f"{int(v)}"
	return ret


def plot_ill_cpg_stat(
	grid: matplotlib.axes.Axes,
	hist_top: matplotlib.axes.Axes,
	hist_right: matplotlib.axes.Axes,
	ill_cpg: pandas.DataFrame,
	ncols: int = 20,
	legend_title: str = None,
):
	# data preparation
	subj = ill_cpg.index.to_list()
	ncpg = ill_cpg.shape[1]
	nrows = len(subj)
	assert nrows < 32
	# calculate share patten between subjects
	basis = 2 ** numpy.arange(nrows).astype(int)
	# count patterns
	pat_count = collections.Counter(
		(basis.reshape(1, -1) @ ill_cpg.values).squeeze()
	)
	sorted_pat = pat_count.most_common()

	# column colors
	col_colors = [dict(facecolor=matplotlib.colors.to_hex(c), edgecolor="none")
		for c in matplotlib.colormaps["turbo"](numpy.linspace(0, 1, ncols))
	]
	# shuffle the colors
	numpy.random.seed(0)
	col_colors = numpy.random.permutation(col_colors).tolist()

	# suffix formatter
	suffix_formatter = matplotlib.ticker.FuncFormatter(_suffix_formatter_func)

	# plot grid
	pat_0_idx = None  # the index of 0, leave None if not found in first-ncols
	circ_r = 0.3
	line_hw = 0.07
	wpad = 0.1
	for col_i, pat in zip(range(ncols), map(lambda x: x[0], sorted_pat)):
		if pat == 0:
			# record the index only
			# don't plot anything except a white column
			pat_0_idx = col_i
			p = matplotlib.patches.Rectangle((col_i, 0), 1, nrows,
				facecolor="#ffffff", edgecolor="none", zorder=2)
			grid.add_patch(p)
			# update col colors
			# pat_0_idx_color = col_colors[col_i]["facecolor"]
			col_colors[pat_0_idx]["facecolor"] = "#ff9d4d"
			continue
		# plot based on pattern
		color = col_colors[col_i]
		ymin, ymax = numpy.nan, numpy.nan
		for row_i, b in enumerate(basis):
			circ_xy = (col_i + 0.5, row_i + 0.5)
			if pat & b:
				# plot black dot
				p = matplotlib.patches.Circle(circ_xy, circ_r, **color, zorder=4)
				grid.add_patch(p)
				ymin = numpy.nanmin([ymin, row_i])  # for line plot later
				ymax = numpy.nanmax([ymax, row_i])  # for line plot later
			# plot white, larger dot
			p = matplotlib.patches.Circle(circ_xy, circ_r + wpad,
				facecolor="#ffffff", edgecolor="none", zorder=2)
			grid.add_patch(p)
		if numpy.isfinite(ymin):
			p = matplotlib.patches.Rectangle(
				(col_i + 0.5 - line_hw, ymin + 0.5), line_hw * 2, ymax - ymin,
				**color, zorder=4)
			grid.add_patch(p)
		p = matplotlib.patches.Rectangle(
			(col_i + 0.5 - line_hw - wpad, 0.5), (line_hw + wpad) * 2,
			nrows - 1, facecolor="#ffffff", edgecolor="none", zorder=2)
		grid.add_patch(p)
	# add a box for all non-zero patterns
	valid_cols = numpy.asarray([i for i in range(ncols) if i != pat_0_idx])
	p = matplotlib.patches.Rectangle((valid_cols.min(), 0),
		valid_cols.max() - valid_cols.min() + 1, nrows,
		clip_on=False, facecolor="none", edgecolor="#404040", zorder=3)
	grid.add_patch(p)
	# misc
	for sp in grid.spines.values():
		sp.set_visible(False)
	grid.tick_params(
		left=False, labelleft=True,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	grid.set_facecolor("#f0f0f0")
	grid.set_xlim(0, ncols)
	grid.set_ylim(0, len(basis))
	grid.set_yticks(numpy.arange(nrows) + 0.5, labels=subj)

	# plot top histogram
	y = numpy.asarray([v[1] for v in sorted_pat[:ncols]])
	x = numpy.arange(ncols) + 0.5
	hist = hist_top.bar(x, y, width=0.9, facecolor="#404040", edgecolor="none")
	hist_top.set_xlim(0, ncols)
	hist_top.set_ylim(0, y.max())
	# mark the 0 pattern
	if pat_0_idx is not None:
		color = col_colors[pat_0_idx]
		hist[pat_0_idx].set(
			facecolor=color["facecolor"],
			edgecolor=color["edgecolor"],
			zorder=2,
		)
	# misc
	for sp in hist_top.spines.values():
		sp.set_visible(False)
	hist_top.tick_params(
		left=True, labelleft=True,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	hist_top.set_ylabel("CpG counts")
	hist_top.yaxis.set_major_formatter(suffix_formatter)

	# plot right horizontal histogram as stacked bars
	hist_right.set_xlim(0, ncpg)
	hist_right.set_ylim(0, nrows)
	left = numpy.zeros(nrows, dtype=float)
	y = numpy.arange(nrows) + 0.5
	for color, (pat, count) in zip(col_colors, sorted_pat):
		# should plot pat = 0 for all rows
		w = [(count if (pat & b) else 0) for b in basis]
		hist_right.barh(y, w, left=left, height=0.9, **color, zorder=2)
		left += w
	# stack up to total ill
	total_ill = ill_cpg.sum(axis=1).values
	hist_right.barh(y, total_ill - left, left=left, height=0.9,
		facecolor="#f0f0f0", edgecolor="none", zorder=2)
	left = total_ill
	_hbar_step(hist_right, left, numpy.arange(nrows + 1),
		color="#404040", linewidth=1.0, zorder=3)
	# stack up to per sample clean
	w = ncpg - pat_count.get(0, 0) - left
	hist_right.barh(y, w, left=left, height=0.9,
		facecolor="#ffdcc0", edgecolor="none", zorder=2)
	left = ncpg - pat_count.get(0, 0)
	# stack up to total cpg
	if 0 in pat_count:
		hist = hist_right.barh(y, pat_count[0], left=left, height=0.9,
			facecolor="#ff9d4d", edgecolor="none", zorder=2)
	# add some lines and labels
	for x in [0, total_ill[-1], ncpg]:
		hist_right.plot([x, x], [nrows + 0.5, nrows + 1.25], clip_on=False,
			linewidth=1.0, color="#404040")
	hist_right.text(total_ill[-1] / 2, nrows + 0.75, "0/1 in\nsample",
		clip_on=False, color="#404040", fontsize=8, zorder=4,
		horizontalalignment="center", verticalalignment="center")
	hist_right.text((ncpg + total_ill[-1]) / 2, nrows + 0.75, "no 0/1 in\nsample",
		clip_on=False, color="#ff9d4d", fontsize=8, zorder=4,
		horizontalalignment="center", verticalalignment="center")
	# misc
	for k, sp in hist_right.spines.items():
		sp.set_visible(k in ["top", "bottom"])
	hist_right.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=True, labelbottom=True,
		top=False, labeltop=False,
	)
	hist_right.set_xlabel("CpG counts")
	hist_right.xaxis.set_major_formatter(suffix_formatter)

	# add 0-pattern annotation
	if pat_0_idx is not None:
		xytext = (
			0.25,
			hist_right.transAxes.inverted().transform(
				hist_top.transAxes.transform((0, 1.2))
			)[1]
		)
		hist_right.text(*xytext, "no 0/1 in all samples", color="#ff9d4d",
			fontsize=10, transform=hist_right.transAxes,
			horizontalalignment="center", verticalalignment="center",
		)
		arrowprops = dict(
			arrowstyle="->", color="#ff9d4d", lw=1.0, shrinkA=55, shrinkB=2,
			connectionstyle="angle,angleA=0,angleB=90,rad=5",
		)
		hist_right.annotate("",
			(ncpg - 0.25 * pat_count[0], nrows), xycoords=hist_right.transData,
			xytext=xytext, textcoords=hist_right.transAxes,
			arrowprops=arrowprops,
		)
		hist_right.annotate("",
			(pat_0_idx + 0.5, pat_count[0]), xycoords=hist_top.transData,
			xytext=xytext, textcoords=hist_right.transAxes,
			arrowprops=arrowprops,
		)

	# legend
	handles = list()
	handles.extend(hist_top.plot([], [], marker="o", linewidth=0, markersize=8,
		color="none", markerfacecolor="#404040", markeredgecolor="none",
		label="has 0/1 in sample"))
	handles.extend(hist_top.plot([], [], marker="o", linewidth=0, markersize=8,
		color="none", markerfacecolor="#ffffff", markeredgecolor="#c0c0c0",
		label="no 0/1 in sample"))
	hist_top.legend(handles=handles, loc=7, bbox_to_anchor=(0.98, 0.50),
		frameon=False, handlelength=0.75, title=legend_title, title_fontsize=14)
	return


def _main():
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# load repl group info
	with open("final.repl.json", "r") as fp:
		repl_group = json.load(fp)
	subj = [v["subject_id"] for v in repl_group]
	repl_to_subj = {}
	for v in repl_group:
		for r in v["replicates"]:
			repl_to_subj[r] = v["subject_id"]

	# load beta data
	beta = pandas.read_csv("final.filtered.beta.tsv", sep="\t", index_col=0)

	# find ill cpg, alredy marked with nan
	ill_cpg = beta.T.isna()
	ill_cpg.index = [repl_to_subj[v] for v in ill_cpg.index]
	# group by index
	ill_cpg = ill_cpg.groupby(ill_cpg.index).any().astype(int)
	# sort rows by row sum
	ill_cpg = ill_cpg.loc[ill_cpg.sum(axis=1).sort_values(ascending=True).index]

	nrows = len(ill_cpg)
	ncols = min(20, len(ill_cpg.columns))

	layout = setup_layout(nrows, ncols, dpi=600)
	figure = layout["figure"]

	plot_ill_cpg_stat(
		layout["grid"],
		layout["hist_top"],
		layout["hist_right"],
		ill_cpg,
		ncols=ncols,
		legend_title=dataset_cfgs[os.path.basename(os.getcwd())].display_name,
	)

	figure.savefig("plot_fig_s14.single.svg", dpi=600)
	figure.savefig("plot_fig_s14.single.png", dpi=600)
	figure.savefig("plot_fig_s14.single.pdf", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
