#!/usr/bin/env python3

import dataclasses
import json

import matplotlib
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot
import mpllayout
import numpy
import threadpoolctl

import pylib


@dataclasses.dataclass
class DepthCateCfg(object):
	dmin: int = None
	dmax: int = None

	def __post_init__(self):
		if self.dmin is not None and self.dmax is not None and self.dmin > self.dmax:
			raise ValueError("Invilid depth range")
		return

	@property
	def display_name(self) -> str:
		if self.dmin == self.dmax == 0:
			ret = "0/Missing"
		elif self.dmin == self.dmax:
			ret = str(self.dmin)
		elif self.dmin is None:
			ret = f"$\\leq$ {self.dmax}"
		elif self.dmax is None:
			ret = f"$\\geq$ {self.dmin}"
		else:
			ret = f"{self.dmin}-{self.dmax}"
		return ret

	def __contains__(self, value: int) -> bool:
		if self.dmin is None:
			ret = value <= self.dmax
		elif self.dmax is None:
			ret = self.dmin <= value
		else:
			ret = self.dmin <= value <= self.dmax
		return ret

	def count_from(self, depth_count: dict[int, int]) -> int:
		return sum(v for k, v in depth_count.items() if k in self)


def load_ill_cpg_surv_data(fname: str) -> dict[str, dict]:
	with open(fname, "r") as fp:
		ret: dict[str] = json.load(fp)
	# convert ill_depth_count keys from str to int
	for v in ret.values():
		v["ill_depth_count"] = {int(k): val
			for k, val in v["ill_depth_count"].items()}
	return ret


def setup_layout(datasets: list[str], ncols: int = 2) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.8,
		right_margin=0.2,
		top_margin=0.4,
		bottom_margin=1.5,
	)
	n_datasets = len(datasets)
	nrows = (n_datasets + ncols - 1) // ncols

	# clock_cov_w = 3.4
	# clock_cov_h = 0.4 * n_datasets
	# clock_cov_top_gap_h = 0.8
	# clock_cov_offset_x = 0.5

	panel_dist_w = 1.8
	panel_dist_h = 0.7
	panel_dist_gap_w = 0.2
	ill_dist_s = 0.9

	block_w = panel_dist_w + panel_dist_gap_w + ill_dist_s
	block_h = max(panel_dist_h, ill_dist_s)
	block_gap_w = 1.0
	block_gap_h = 0.7

	# ill cpg clock coverate
	# clock_cov = lc.add_frame("clock_cov_ill")
	# clock_cov.set_anchor("bottomleft", offsets=(clock_cov_offset_x, 0))
	# clock_cov.set_size(clock_cov_w, clock_cov_h)

	# axes complex of distr curve and pie
	for i, v in enumerate(datasets):
		ri = nrows - (i // ncols) - 1
		ci = i % ncols

		panel_dist = lc.add_frame(f"{v}_panel_dist")
		panel_dist.set_anchor("bottomleft",
			offsets=(
				(block_w + block_gap_w) * ci,
				(block_h + block_gap_h) * ri,
			)
		)
		panel_dist.set_size(panel_dist_w, panel_dist_h)

		ill_dist = lc.add_frame(f"{v}_ill_dist")
		ill_dist.set_anchor("topleft",
			ref_frame=panel_dist,
			ref_anchor="topright",
			offsets=(panel_dist_gap_w, 0)
		)
		ill_dist.set_size(ill_dist_s, ill_dist_s)

	layout = lc.create_figure_layout()
	layout["nrows"] = nrows
	layout["ncols"] = ncols

	return layout


def _plot_curve_area(axes: matplotlib.axes.Axes,
	x: numpy.ndarray, y: numpy.ndarray, *,
	color: str,
	base: float = 0.0,
):
	y /= y.max()
	axes.plot(x, y + base, clip_on=False, linewidth=1.0, color=color, zorder=5)
	axes.fill_between(x, base, y + base, clip_on=False, edgecolor="none",
		facecolor=color + "40", zorder=4,
	)
	axes.axhline(base, linewidth=0.5, color=color, zorder=5)
	return


def _plot_ill_cpg_stat_panel_dist(axes: matplotlib.axes.Axes, data: dict[str], *,
	dataset_cfg: pylib.DatasetCfg,
	connect_axes: matplotlib.axes.Axes,
):
	edges = numpy.array(data["beta_hist_bins"])
	pos = (edges[:-1] + edges[1:]) / 2
	density = numpy.array(data["beta_hist"])

	# plot full value curve
	_plot_curve_area(axes, pos, density, color=dataset_cfg.color, base=0.0)

	# add red boxes
	for v in [0, -1]:
		p = matplotlib.patches.Rectangle((pos[v] - 0.02, -0.08),
			0.04, density[v] + 0.16,
			clip_on=False, zorder=6, linestyle="--", linewidth=0.7,
			edgecolor="#ff0000c0", facecolor="none",
		)
		axes.add_patch(p)

	# add connections
	p = matplotlib.patches.ConnectionPatch(
		xyA=(pos[0] + 0.02, -0.08), coordsA=axes.transData,
		xyB=(pos[-1] - 0.02, -0.08), coordsB=axes.transData,
		edgecolor="#ff000080", facecolor="none",
	)
	axes.add_artist(p)
	p = matplotlib.patches.ConnectionPatch(
		xyA=(pos[0] + 0.02, density[0] + 0.08), coordsA=axes.transData,
		xyB=(pos[-1] - 0.02, density[-1] + 0.08), coordsB=axes.transData,
		edgecolor="#ff000080", facecolor="none",
	)
	axes.add_artist(p)
	p = matplotlib.patches.ConnectionPatch(
		xyA=(pos[-1] + 0.02, -0.08), coordsA=axes.transData,
		xyB=(-0.04, -0.04), coordsB=connect_axes.transAxes,
		edgecolor="#ff000080", facecolor="none",
	)
	axes.add_artist(p)
	p = matplotlib.patches.ConnectionPatch(
		xyA=(pos[-1] + 0.02, density[-1] + 0.08), coordsA=axes.transData,
		xyB=(-0.04, 1.04), coordsB=connect_axes.transAxes,
		edgecolor="#ff000080", facecolor="none",
	)
	axes.add_artist(p)

	# misc
	axes.tick_params(
		left=False, labelleft=False,
	)
	axes.grid(axis="x", color="#e0e0e0")
	axes.set_xlabel("Beta")
	axes.set_ylabel("P.D.")
	title = dataset_cfg.display_name
	if dataset_cfg.technology not in title:
		title += f" [{dataset_cfg.technology}]"
	axes.set_title(title)

	return


def _plot_ill_cpg_stat_ill_dist(axes: matplotlib.axes.Axes, data: dict[str], *,
	depth_cate_cfgs: list[DepthCateCfg],
	add_legend: bool = False,
):
	cmap = matplotlib.colormaps["viridis"]

	# count depth in each category
	depth_count = data["ill_depth_count"]
	depth_total = sum(depth_count.values())

	cate_depth = [cfg.count_from(depth_count) for cfg in depth_cate_cfgs]
	cate_total = sum(cate_depth)

	# plot for each names depth
	cur_theta = -(cate_total / depth_total * 180)
	handles = list()
	for i, d in enumerate(cate_depth):
		cfg = depth_cate_cfgs[i]
		_theta = d / depth_total * 360
		facecolor = "#d0d0d0" if (i == 0) else cmap(i / (len(depth_cate_cfgs) - 1))
		# pyplot
		p = matplotlib.patches.Wedge((0, 0), 1, cur_theta, cur_theta + _theta,
			clip_on=False, label=cfg.display_name, zorder=5, linewidth=0.5,
			edgecolor="#ffffff", facecolor=facecolor,
		)
		axes.add_patch(p)
		cur_theta += _theta
		handles.append(p)

	# add other if necessary
	if cate_total < depth_total:
		_theta = (depth_total - cate_total) / depth_total * 360
		label = f"> {depth_cate_cfgs[-1].dmax}"
		p = matplotlib.patches.Wedge((0, 0), 1, cur_theta, cur_theta + _theta,
			clip_on=False, label=label, zorder=5, linewidth=0.5,
			edgecolor="#000000", facecolor="#ffffff",
		)
		axes.add_patch(p)
		handles.append(p)

	# add center circle and text
	p = matplotlib.patches.Circle((0, 0), 0.5, clip_on=False, edgecolor="none",
		facecolor="#ffffff", zorder=10)
	axes.add_patch(p)
	axes.text(0, 0,
		f"{data["ill_frac"] * 100:.1f}%",
		fontweight="bold", zorder=11,
		horizontalalignment="center", verticalalignment="center",
	)

	# add a 3-side box to complete the connection
	line = matplotlib.lines.Line2D([-0.04, 1.04, 1.04, -0.04],
		[-0.04, -0.04, 1.04, 1.04], clip_on=False, transform=axes.transAxes,
		linestyle="-", linewidth=1.0, color="#ff000080"
	)
	axes.add_artist(line)

	# legend
	if add_legend:
		axes.legend(handles=handles, loc=1, bbox_to_anchor=[1.02, -0.36],
			frameon=True, handlelength=0.75, ncol=3,  # title="Depth",
			fontsize=10, title="Depth ranges", title_fontsize=12,
		)

	# misc
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	axes.set_xlabel("Depths of\n0&1 betas")
	axes.set_xlim(-1, 1)
	axes.set_ylim(-1, 1)
	return


def _plot_ill_cpg_stat(*,
	panel_dist_axes: matplotlib.axes.Axes,
	ill_dist_axes: matplotlib.axes.Axes,
	data: dict[str],
	dataset_cfg: pylib.DatasetCfg,
	depth_cate_cfgs: list[DepthCateCfg],
	add_ill_dist_legend: bool = False,
):
	# plot beta value distribution of full panel
	_plot_ill_cpg_stat_panel_dist(panel_dist_axes, data,
		dataset_cfg=dataset_cfg,
		connect_axes=ill_dist_axes,
	)

	# plot depth distribution of ill cpgs
	_plot_ill_cpg_stat_ill_dist(ill_dist_axes, data,
		depth_cate_cfgs=depth_cate_cfgs,
		add_legend=add_ill_dist_legend,
	)

	return


def _plot_ill_cpg_survey(layout: dict, datasets: list[str], clocks: list[str],
	dataset_cfgs: pylib.DatasetCfgLib,
	depth_cate_cfgs: list[DepthCateCfg],
):
	n_datasets = len(datasets)
	ncols = layout["ncols"]

	# load ill cpg surv data
	data = load_ill_cpg_surv_data("ill_cpg_stat.json")

	# plot
	if n_datasets > ncols:
		legend_idx = (n_datasets // ncols) * ncols - 1
	else:
		legend_idx = n_datasets - 1
	for i, ds in enumerate(datasets):
		_plot_ill_cpg_stat(
			panel_dist_axes=layout[f"{ds}_panel_dist"],
			ill_dist_axes=layout[f"{ds}_ill_dist"],
			data=data[ds],
			dataset_cfg=dataset_cfgs[ds],
			depth_cate_cfgs=depth_cate_cfgs,
			add_ill_dist_legend=(i == legend_idx),
		)

	return


def _main():
	# configs
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST",
		"RB_SYF", "GSE144691", "GSE86832", "BUCCAL_TWIST"]
	depth_cate_cfgs = [
		DepthCateCfg(dmin=0, dmax=0),
		DepthCateCfg(dmin=1, dmax=2),
		DepthCateCfg(dmin=3, dmax=5),
		DepthCateCfg(dmin=6, dmax=10),
		DepthCateCfg(dmin=11, dmax=20),
		DepthCateCfg(dmin=21, dmax=None),
	]

	# load data
	dataset_cfgs = pylib.DatasetCfgLib.from_json()
	clocks = pylib.clock.load_clocks()

	# plot
	layout = setup_layout(datasets)
	figure = layout["figure"]

	_plot_ill_cpg_survey(layout, datasets=datasets,
		clocks=clocks,
		dataset_cfgs=dataset_cfgs,
		depth_cate_cfgs=depth_cate_cfgs,
	)

	figure.savefig("fig_s8.svg", dpi=600)
	matplotlib.pyplot.close(figure)

	return


if __name__ == "__main__":
	with threadpoolctl.threadpool_limits(limits=24, user_api="blas"):
		_main()
