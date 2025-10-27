#!/usr/bin/env python3

import dataclasses
import functools
import pickle

import matplotlib
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.lines
import mpllayout
import numpy
import pandas

import pylib

matplotlib.rcParams["font.size"] = 8


@dataclasses.dataclass
class SimMethodCfg(object):
	key: str
	file: str
	xlabel: str
	ruler: float = None
	xmax: float = 200
	title: str = None

	@functools.cached_property
	def stat(self) -> dict[str, dict[str, pandas.DataFrame]]:
		pylib.logger.info(f"loading data from {self.file}")
		with open(self.file, "rb") as fp:
			ret = pickle.load(fp)
		return ret


@dataclasses.dataclass
class SimSeriesCfg(object):
	key: str
	color: str
	display_name: str
	unit: str

	@property
	def ylabel(self) -> str:
		return f"{self.display_name} ({self.unit})"


class ShadedLineHandler(matplotlib.legend_handler.HandlerLine2D):
	def create_artists(self, legend, orig_handle, xdescent, ydescent,
		width, height, fontsize, transform,
	) -> list:
		artists = super().create_artists(legend, orig_handle,
			xdescent, ydescent, width, height, fontsize, transform
		)
		rect_h = (artists[0].get_ydata()[0] - ydescent) * 2

		# create the shade artist
		artist = matplotlib.patches.Rectangle((0, ydescent), width, rect_h,
			edgecolor="none", facecolor=orig_handle.get_color() + "20",
		)
		artists.append(artist)
		return artists


def setup_layout(*, sim_method_cfgs: list[SimMethodCfg],
	sim_series_cfgs: list[SimSeriesCfg],
):
	lc = mpllayout.LayoutCreator(
		left_margin=0.7,
		right_margin=0.2,
		bottom_margin=1.2,
		top_margin=0.5,
	)

	# depth sims
	n_sim_methods = len(sim_method_cfgs)
	n_sim_series = len(sim_series_cfgs)

	sim_w = 2.0
	sim_h = 2.0
	sim_gap_h = 0.1
	sim_meth_w = sim_w
	sim_meth_gap_w = 0.7

	for mi, method_cfg in enumerate(sim_method_cfgs):
		for si, series_cfg in enumerate(sim_series_cfgs):
			axes = lc.add_frame(f"axes_{series_cfg.key}_{method_cfg.key}")
			axes.set_anchor("bottomleft", offsets=(
				(sim_meth_w + sim_meth_gap_w) * mi,
				(sim_h + sim_gap_h) * (n_sim_series - si - 1),
			))
			axes.set_size(sim_w, sim_h)

	layout = lc.create_figure_layout()

	return layout


def _plot_depth_sim_stat_single(axes: matplotlib.axes.Axes,
	sim_method_cfg: SimMethodCfg, *,
	datasets: list[str], dataset_cfgs: dict[str, pylib.DatasetCfg],
	sim_series_cfg: SimSeriesCfg, n_list: list[int] = None,
	add_legend: bool = False, show_xlabel: bool = False, show_title: bool = False,
) -> None:

	stat = sim_method_cfg.stat
	if n_list is None:
		n_list = stat[datasets[0]][sim_series_cfg.key].index.tolist()

	handles = list()
	for ds in datasets:
		vals = stat[ds][sim_series_cfg.key].values.astype(float)
		lb = numpy.quantile(vals, 0.25, axis=1)
		ub = numpy.quantile(vals, 0.75, axis=1)
		mean = vals.mean(axis=1)

		ds_cfg = dataset_cfgs[ds]
		p = axes.plot(n_list, mean, linewidth=1.0, color=ds_cfg.color, zorder=10,
			label=ds_cfg.display_name)[0]
		handles.append(p)
		axes.fill_between(n_list, lb, ub, zorder=5,
			edgecolor="none", facecolor=ds_cfg.color + "20",
		)

	if sim_method_cfg.ruler is not None:
		axes.axvline(sim_method_cfg.ruler, color="#e36363",
			linestyle="-", linewidth=1.0, zorder=8)
		# assume ylim is not adjusted
		text_y = axes.transData.inverted().transform(
			axes.transAxes.transform((0, 0.95))
		)[1]
		axes.text(sim_method_cfg.ruler, text_y, f" {sim_method_cfg.ruler:d}",
			color="#e36363", fontsize=10, zorder=8,
			horizontalalignment="left", verticalalignment="top",
		)

	# legend
	if add_legend:
		handler_map = {matplotlib.lines.Line2D: ShadedLineHandler()}
		legend = axes.legend(handles=handles, handler_map=handler_map,
			loc=2, bbox_to_anchor=(0.05, -0.30), ncol=4, handlelength=1.0,
		)

	# misc
	axes.grid(linewidth=0.5, color="#d0d0d0")
	axes.tick_params(
		left=True, labelleft=True,
		right=False, labelright=False,
		bottom=True, labelbottom=show_xlabel,
		top=False, labeltop=False,
	)
	axes.set_xlim(0, min(max(n_list), sim_method_cfg.xmax))
	if show_xlabel:
		axes.set_xlabel(sim_method_cfg.xlabel, fontsize=10)
	axes.set_ylabel(sim_series_cfg.ylabel, fontsize=10)
	if show_title and (sim_method_cfg.title is not None):
		axes.set_title(sim_method_cfg.title, fontsize=10)

	return


def _plot_depth_sim_stat(layout: dict[str], method_cfgs: list[SimMethodCfg],
	*, datasets: list[str], dataset_cfgs: dict[str, pylib.DatasetCfg],
	sim_series_cfgs: list[SimSeriesCfg], n_list: list[int] = None,
	skip: bool = False,
) -> None:
	if skip:
		pylib.logger.info("skipping plotting depth sim stat")
		return

	n_series = len(sim_series_cfgs)

	for method_cfg in method_cfgs:
		for si, series_cfg in enumerate(sim_series_cfgs):
			_plot_depth_sim_stat_single(
				layout[f"axes_{series_cfg.key}_{method_cfg.key}"],
				method_cfg,
				datasets=datasets,
				dataset_cfgs=dataset_cfgs,
				sim_series_cfg=series_cfg,
				n_list=n_list,
				add_legend=(si == n_series - 1) and (method_cfg == method_cfgs[0]),
				show_xlabel=(si == n_series - 1),
				show_title=(si == 0),
			)
	return


def _main():
	# configs
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	sim_method_cfgs = [
		SimMethodCfg(key="binomial_stoc", xlabel="Depth",
			ruler=20, xmax=100, file="binomial_stoc.stat.pkl",
			title="Effect on\nbinomial stochasticity",
		),
		SimMethodCfg(key="depth_round", xlabel="Depth",
			ruler=10, xmax=50, file="beta_resolution.stat.pkl",
			title="Effect on\nbeta value resolution",
		),
	]
	sim_series_cfgs = [
		SimSeriesCfg(key="mae", color="#3F6EE5",
			display_name="MAE", unit="years"),
		SimSeriesCfg(key="repl_mad", color="#E5763F",
			display_name="RD", unit="years"),
	]

	# load data
	dataset_cfgs = pylib.DatasetCfgLib.from_json()
	clock_cfgs = pylib.clock_cfg.ClockCfgLib.from_json()

	# plot
	layout = setup_layout(
		sim_series_cfgs=sim_series_cfgs,
		sim_method_cfgs=sim_method_cfgs,
	)

	figure = layout["figure"]

	_plot_depth_sim_stat(layout, sim_method_cfgs,
		datasets=datasets, dataset_cfgs=dataset_cfgs,
		sim_series_cfgs=sim_series_cfgs,
		skip=False,
	)

	figure.savefig("fig_s13.svg", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
