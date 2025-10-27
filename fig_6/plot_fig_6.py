#!/usr/bin/env python3

import collections
import dataclasses
import functools
import pdb
import pickle
from typing import Callable, Self, Sequence
import types

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
import scipy
import sklearn.metrics

import pylib

matplotlib.rcParams["font.size"] = 8


class StratSeleKeyCfgLib(set):
	@dataclasses.dataclass
	class KeyCfg(object):
		depth_filtering: int | None
		imputation: bool
		transfer_learning: bool

		# def __len__(self) -> int:
		# return 3

		def __post_init__(self):
			if (self.depth_filtering is None) or (self.depth_filtering < 0):
				self.depth_filtering = 0
			return

		def to_tuple(self) -> tuple:
			return (self.depth_filtering, self.imputation, self.transfer_learning)

		def __hash__(self) -> int:
			return hash(self.to_tuple())

		@property
		def order_key(self) -> int:
			ret = -int(self.transfer_learning) * 100 \
				- int(self.imputation) * 10 \
				- bool(self.depth_filtering)
			return ret

	def add_new(self, **kw) -> KeyCfg:
		new = self.KeyCfg(**kw)
		self.add(new)
		return new

	@property
	def key_field_values(self) -> dict[str, set]:
		ret = collections.defaultdict(set)
		for cfg in self:
			for field in dataclasses.fields(cfg):
				name = field.name
				ret[name].add(getattr(cfg, name))
		return ret


class StratSeleKeyFieldList(list["StratSeleKeyFieldList.KeyFieldCfg"]):
	@dataclasses.dataclass
	class KeyFieldCfg(object):
		key: str
		full_name: str
		short_name: str
		color: str
		# return if the field is 'active' given a field value
		is_active_in: Callable[[Self, StratSeleKeyCfgLib.KeyCfg], bool] = \
			lambda self, v: bool(getattr(v, self.key))
		# return the representative label given a field value
		get_active_label: Callable[[Self, StratSeleKeyCfgLib.KeyCfg], str | None] = \
			lambda self, v: self.short_name

		def __post_init__(self):
			self.is_active_in = types.MethodType(self.is_active_in, self)
			self.get_active_label = types.MethodType(self.get_active_label, self)
			return

		def get_label(self, v: StratSeleKeyCfgLib.KeyCfg) -> str | None:
			return self.get_active_label(v) if self.is_active_in(v) else None

	def get_field_labels(self, cfg: StratSeleKeyCfgLib.KeyCfg) -> list[str]:
		ret = list()
		for field in self:
			if (v := field.get_label(cfg)) is not None:
				ret.append(v)
		return ret


@dataclasses.dataclass
class StratSeleMetricCfg(object):
	key: str
	display_name: str
	unit: str = None

	@property
	def label(self) -> str:
		if self.unit is None:
			return self.display_name
		else:
			return f"{self.display_name} ({self.unit})"


@dataclasses.dataclass
class BestStratSeriesCfg(object):
	key: tuple
	display_name: str
	marker: str = "o"
	color_override: str = None
	base_zorder: int = 10
	text_xy: tuple[float, float] = (0, 0)
	text_props: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class BestStratDataCfg(object):
	key: str
	file: str

	@functools.cached_property
	def data(self) -> dict:
		with open(self.file, "rb") as fp:
			ret = pickle.load(fp)
		return ret


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


def _strat_sele_load_results(fname: str,
) -> tuple[StratSeleKeyCfgLib, list[str], dict[StratSeleKeyCfgLib.KeyCfg, pandas.DataFrame]]:
	with open(fname, "rb") as fp:
		raw = pickle.load(fp)

	key_fields = raw["key_fields"]

	key_cfgs = StratSeleKeyCfgLib()
	res = dict()

	for k, df in raw["res"].items():
		cfg = key_cfgs.add_new(**dict(zip(key_fields, k)))
		res[cfg] = df

	return key_cfgs, raw["clocks"], res


def setup_layout(*, downsample_series: list[str],
	strat_sele_metric_cfgs: list[StratSeleMetricCfg],
	strat_sele_key_cfgs: StratSeleKeyCfgLib,
	best_strat_data_cfgs: list[BestStratDataCfg],
	best_strat_clocks: list[str], best_strat_ncols: int = 2,
):
	lc = mpllayout.LayoutCreator(
		left_margin=0.7,
		right_margin=0.2,
		bottom_margin=1.2,
		top_margin=0.5,
	)

	# best strat
	n_best_strat_clocks = len(best_strat_clocks)
	best_strat_nrows = (n_best_strat_clocks + best_strat_ncols - 1) \
		// best_strat_ncols

	best_strat_w = 1.6
	best_strat_h = 1.6
	best_strat_gap_w = 0.2
	best_strat_gap_h = 0.4
	best_strat_ds_w = best_strat_w * best_strat_ncols + \
		best_strat_gap_w * (best_strat_ncols - 1)
	best_strat_ds_h = best_strat_h * best_strat_nrows + \
		best_strat_gap_h * (best_strat_nrows - 1)
	best_strat_ds_gap_w = 1.0

	for di, ds in enumerate(best_strat_data_cfgs):
		for mi, clock in enumerate(best_strat_clocks):
			ri = best_strat_nrows - (mi // best_strat_ncols) - 1
			ci = mi % best_strat_ncols
			axes = lc.add_frame(f"best_strat_{ds.key}_{clock}")
			axes.set_anchor("bottomleft", offsets=(
				(best_strat_ds_w + best_strat_ds_gap_w) *
				 di + (best_strat_w + best_strat_gap_w) * ci,
				(best_strat_h + best_strat_gap_h) * ri,
			))
			axes.set_size(best_strat_w, best_strat_h)

	# strat_select
	strat_sele_base_w = 0.5
	strat_sele_base_h = best_strat_ds_h + 0.8

	strat_sele_key_field_cfg_sp = 0.25
	strat_sele_key_field_elem_sp = 0.25
	strat_sele_key_ncol = len(next(iter(strat_sele_key_cfgs)).to_tuple())
	strat_sele_key_nrow = len(strat_sele_key_cfgs)

	strat_sele_key_w = strat_sele_key_field_elem_sp * strat_sele_key_ncol
	strat_sele_key_h = strat_sele_key_field_cfg_sp * strat_sele_key_nrow
	strat_sele_key_gap = 0.1
	strat_sele_box_w = 2.8
	strat_sele_box_h = strat_sele_key_h

	for i, cfg in enumerate(strat_sele_metric_cfgs):
		strat_sele_key = lc.add_frame(f"strat_sele_key_{cfg.key}")
		strat_sele_key.set_size(strat_sele_key_w, strat_sele_key_h)
		strat_sele_key.set_anchor("bottomleft", offsets=(
			strat_sele_base_w,
			strat_sele_base_h,
		))
		strat_sele_box = lc.add_frame(f"strat_sele_box_{cfg.key}")
		strat_sele_box.set_size(strat_sele_box_w, strat_sele_box_h)
		strat_sele_box.set_anchor("bottomleft", strat_sele_key, "bottomright",
		    offsets=(strat_sele_key_gap, 0))

	# model downsample
	n_dsamp = len(downsample_series)

	dsamp_base_w = 0
	dsamp_base_h = best_strat_ds_h + 3.4
	dsamp_w = 3.6
	dsamp_h = 1.8
	dsamp_gap_w = 0.6

	for si, series in enumerate(downsample_series):
		axes = lc.add_frame(f"dsamp_{series}")
		axes.set_anchor("bottomleft", offsets=(
			dsamp_base_w + (dsamp_w + dsamp_gap_w) * si,
			dsamp_base_h,
		))
		axes.set_size(dsamp_w, dsamp_h)

	layout = lc.create_figure_layout()

	return layout


def _plot_downsample(layout: dict[str], clocks: list[str], *,
	clock_cfgs=pylib.clock_cfg.ClockCfgLib,
	skip: bool = False,
) -> None:
	if skip:
		pylib.logger.info("skipping plotting downsample")
		return

	# load data
	pred_mae = pandas.read_csv("downsampling_mae.csv",
		index_col=0)
	with open("downsampling_20x_ratio.pkl", "rb") as fp:
		frac_20x_sites = pickle.load(fp)

	# plot mae scatter
	handles = list()
	axes: matplotlib.axes.Axes = layout["dsamp_mae"]
	for c in clocks:
		clock_cfg = clock_cfgs[c]
		y = pred_mae[c].values
		mask = numpy.isfinite(y)
		x = pred_mae.index.values[mask]
		y = y[mask]
		# line
		axes.plot(x, y, linewidth=1.0, color=clock_cfg.color, zorder=10)
		# point background
		axes.scatter(x, y, s=30, marker=clock_cfg.marker, linewidth=0.5,
			edgecolors="none", facecolors="#ffffff", zorder=11,
		)
		# point foregound overlay
		h = axes.scatter(x, y, s=30, marker=clock_cfg.marker, linewidth=0.5,
			edgecolors="#000000", facecolors=clock_cfg.color + "80",
			zorder=12, label=clock_cfg.name,
		)
		handles.append(h)
	# legend
	legend = axes.legend(handles=handles, loc=1, bbox_to_anchor=(0.98, 0.98),
		handlelength=1.5, ncol=2,
	)
	# misc
	axes.grid(linewidth=0.5, color="#d0d0d0")
	axes.set_xlabel("Downsampling depth", fontsize=10)
	axes.set_ylabel("MAE (years)", fontsize=10)

	# plot frac 20x
	handles.clear()
	axes: matplotlib.axes.Axes = layout["dsamp_perc_20x"]
	axes.sharex(layout["dsamp_mae"])
	for c in clocks:
		clock_cfg = clock_cfgs[c]

		mask = numpy.isfinite(pred_mae[c].values)
		xv_dict = frac_20x_sites[c]
		keys = sorted(xv_dict.keys())
		x = pred_mae.index.values[mask][keys]
		y_mean = numpy.asarray([numpy.mean(xv_dict[k]) for k in keys])
		y_std = numpy.asarray([numpy.std(xv_dict[k]) for k in keys])
		y_mean *= 100
		y_std *= 100

		# line
		axes.plot(x, y_mean, linewidth=1.0, color=clock_cfg.color, zorder=10)
		axes.errorbar(x, y_mean, yerr=y_std, linewidth=1.0,
			elinewidth=1.0, capsize=2.0, capthick=1.0,
			fmt="none", ecolor=clock_cfg.color, zorder=10,
		)
		# point background
		axes.scatter(x, y_mean, s=30, marker=clock_cfg.marker, linewidth=0.5,
			edgecolors="none", facecolors="#ffffff", zorder=11,
		)
		# point foregound overlay
		h = axes.scatter(x, y_mean, s=30, marker=clock_cfg.marker, linewidth=0.5,
			edgecolors="#000000", facecolors=clock_cfg.color + "80",
			zorder=12, label=clock_cfg.name,
		)
		handles.append(h)
	# legend
	legend = axes.legend(handles=handles, loc=4, bbox_to_anchor=(0.98, 0.02),
		handlelength=1.5, ncol=1,
	)
	# misc
	axes.grid(linewidth=0.5, color="#d0d0d0")
	axes.set_xlabel("Downsampling depth", fontsize=10)
	axes.set_ylabel(r"CpG coverage $\geq$20X (%)", fontsize=10)

	return


def _strat_sele_calc_stats(data: dict[tuple, pandas.DataFrame], clocks: Sequence[str],
	*, repl_group: pylib.dataset.ReplGroup = None,
) -> dict[tuple, pandas.DataFrame]:
	ret = dict()
	for k, df in data.items():
		age_pred_res = pylib.age_pred_res.AgePredRes(df)
		ret[k] = age_pred_res.calc_stat(clocks, repl_group=repl_group)
	return ret


def _sort_stat_keys(stats: dict[tuple, pandas.DataFrame], metric: StratSeleMetricCfg, *,
	reverse: bool = False,
) -> list[StratSeleKeyCfgLib.KeyCfg]:

	ret = sorted(stats.keys(), key=lambda k: k.order_key,
		# key=lambda k: numpy.median(stats[k][metric.key]),
		reverse=reverse)

	return ret


def _strat_sele_plot_stats(box_axes: matplotlib.axes.Axes,
	key_field_axes: matplotlib.axes.Axes, *,
	stats: dict[tuple, pandas.DataFrame],
	metric: StratSeleMetricCfg,
	key_order=list[StratSeleKeyCfgLib.KeyCfg],
	key_fields: StratSeleKeyFieldList,
	highlight_clock_cfgs: pylib.ClockCfgLib,
	box_ec: str = "#000000",
	box_fc: str = "#c0c0c0",
) -> None:

	n_keys = len(key_order)
	n_key_fields = len(key_fields)

	# plot key field patches
	for i, key in enumerate(key_order):
		for j, field in enumerate(key_fields):
			if field.is_active_in(key):
				p = matplotlib.patches.Circle((j + 0.5, i + 0.5), 0.3,
					linewidth=0.5, edgecolor="#000000", facecolor=field.color,
					zorder=10
				)
				key_field_axes.add_patch(p)

	# add horizontal arrows
	for i in range(n_keys):
		a = matplotlib.patches.FancyArrowPatch(
			(-0.2, i + 0.5), (n_key_fields + 0.4, i + 0.5),
			arrowstyle='-|>', mutation_scale=10, linewidth=2.0, color="#c0c0c0",
			clip_on=False, zorder=5,
		)
		key_field_axes.add_patch(a)

	# misc
	key_field_axes.tick_params(
		left=False, labelleft=True,
		right=False, labelright=False,
		bottom=True, labelbottom=True,
		top=False, labeltop=False,
	)
	for sp in key_field_axes.spines.values():
		sp.set_visible(False)
	key_field_axes.set(
		xlim=(0, len(key_fields)),
		ylim=(0, n_keys),
	)
	key_field_axes.set_xticks(numpy.arange(n_key_fields) + 0.5, fontsize=8,
		rotation=90,
		labels=[f.short_name for f in key_fields],
	)
	key_field_axes.set_yticks(numpy.arange(n_keys) + 0.5, fontsize=8, labels=[
		(("+").join(key_fields.get_field_labels(k)) or "Direct") for k in key_order
	])

	# plot boxes
	for i, key in enumerate(key_order):
		vals = stats[key][metric.key].values

		box = box_axes.boxplot(vals,
			orientation="horizontal", positions=[i + 0.5],
			widths=0.7, showfliers=False, patch_artist=True,
			boxprops=dict(linewidth=0.5, edgecolor=box_ec, facecolor=box_fc),
			medianprops=dict(linewidth=1.0, color=box_ec),
			whiskerprops=dict(linewidth=0.5, color=box_ec),
			capprops=dict(linewidth=0.5, color=box_ec),
			capwidths=0.6,
		)

	# add clock scatters
	clocks = sorted(stats[key_order[0]].index)
	y = numpy.arange(n_keys) + 0.5

	handles = list()
	for c in clocks:
		x = [stats[key].loc[c, metric.key] for key in key_order]

		if c in highlight_clock_cfgs:
			cfg = highlight_clock_cfgs[c]
			marker = cfg.marker
			edgecolor = "#000000"
			facecolor = cfg.color + "a0"
			zorder = 12
			label = cfg.name
		else:
			marker = "."
			edgecolor = "none"
			facecolor = box_ec
			zorder = 10
			label = None

		h = box_axes.scatter(x, y, marker=marker, s=24, linewidth=0.5,
			edgecolors=edgecolor, facecolors=facecolor,
			zorder=zorder, label=label,
		)

		if c in highlight_clock_cfgs:
			handles.append(h)

	# clock legend
	box_axes.legend(handles=handles, loc=6, bbox_to_anchor=(1.00, 0.50),
		frameon=True, ncol=3, handlelength=0.75, title_fontsize=10,
	)

	# misc
	box_axes.grid(linewidth=0.5, color="#d0d0d0", axis="x")
	box_axes.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=True, labelbottom=True,
		top=False, labeltop=False,
	)
	box_axes.set_xlim(0, None)
	box_axes.set_ylim(0, n_keys)
	box_axes.set_yticks(numpy.arange(n_keys) + 0.5)
	box_axes.set_xlabel(metric.label, fontsize=10)

	return


def _plot_best_strat_dataset_clock_single(axes: matplotlib.axes.Axes,
	data: dict, *, data_cfg: BestStratDataCfg, clock: str,
	series_cfgs: list[BestStratSeriesCfg], dataset_cfg: pylib.DatasetCfg,
	clock_cfg: pylib.clock_cfg.ClockCfg, add_legend: bool = False,
	show_xticks: bool = False, show_yticks: bool = False,
	show_xlabel: bool = False, show_ylabel: bool = False,
	add_dataset_label: bool = False,
) -> None:

	for i, series in enumerate(series_cfgs):
		res = data[series.key]["res"]
		x = res.data["age"]
		y = res.data[clock]
		if series.color_override is not None:
			edgecolor = series.color_override
			facecolor = series.color_override
		else:
			edgecolor = "#000000"
			facecolor = clock_cfg.color + "80"
		# plot scatter
		axes.scatter(x, y, s=20, marker=series.marker, linewidths=0.5,
			edgecolors=edgecolor, facecolors=facecolor,
			zorder=series.base_zorder,
		)

		# add text
		linreg = scipy.stats.linregress(x, y)
		t = ("[{}]\nR={:.2f}\np={:.2e}\nMAE={:.1f} yrs.").format(
			series.display_name, linreg.rvalue, linreg.pvalue,
			sklearn.metrics.mean_absolute_error(x, y)
		)
		axes.text(*series.text_xy, t, fontsize=8,
			transform=axes.transAxes, zorder=series.base_zorder + 1,
			**series.text_props,
		)

	# add diag line
	line = matplotlib.lines.Line2D([0, 150], [0, 150], linewidth=1.0,
		color="#000000", linestyle="--")
	axes.add_artist(line)

	# legend
	if add_legend:
		handles = list()
		for series in series_cfgs:
			if series.color_override is not None:
				edgecolor = series.color_override
				facecolor = series.color_override
			else:
				edgecolor = "#000000"
				facecolor = "#e0e0e0"
			h = axes.scatter([], [], s=20, marker=series.marker, linewidths=0.5,
				edgecolors=edgecolor, facecolors=facecolor,
				label=series.display_name,
			)
			handles.append(h)
		legend = axes.legend(handles=handles, loc=9, bbox_to_anchor=(1.32, -0.18),
			handlelength=0.75, ncol=2, title="Improvement strategies",
		)

	# misc
	axes.tick_params(
		left=True, labelleft=show_yticks,
		right=False, labelright=False,
		bottom=True, labelbottom=show_xticks,
		top=False, labeltop=False,
	)

	axes.grid(linewidth=0.5, color="#d0d0d0")
	if show_xlabel:
		axes.text(-0.05, -0.25, "Chronological age (years)", fontsize=10,
			transform=axes.transAxes,
			horizontalalignment="center", verticalalignment="center",
		)
	if show_ylabel:
		axes.text(-0.28, 0.5, "Predicted age (years)", fontsize=10,
			transform=axes.transAxes, rotation=90,
			horizontalalignment="center", verticalalignment="center",
		)
	axes.set_title(clock_cfg.name, fontsize=10)
	if add_dataset_label:
		label = f"{dataset_cfg.display_name} (N={dataset_cfg.n_samples:d})"
		axes.text(-0.05, -0.35, label, fontsize=12,
			fontweight="bold", transform=axes.transAxes,
			horizontalalignment="center", verticalalignment="top",
		)

	return


def _plot_best_strat_dataset_single(layout: dict[str], data_cfg: BestStratDataCfg,
	*, dataset_cfgs: pylib.DatasetCfgLib, clocks: list[str],
	series_cfgs: list[BestStratSeriesCfg],
) -> None:
	data = data_cfg.data
	# make plot data
	plot_data = dict()
	for series in series_cfgs:
		age_pred_res = pylib.AgePredRes(data["res"][series.key])
		plot_data[series.key] = dict(
			res=age_pred_res,
			stat=age_pred_res.calc_stat(clocks)
		)
	clock_cfgs = pylib.clock_cfg.ClockCfgLib.from_json()

	# share x and y axes
	for clock in clocks[1:]:
		layout[f"best_strat_{data_cfg.key}_{clock}"].sharex(
			layout[f"best_strat_{data_cfg.key}_{clocks[0]}"]
		)
		layout[f"best_strat_{data_cfg.key}_{clock}"].sharey(
			layout[f"best_strat_{data_cfg.key}_{clocks[0]}"]
		)

	n_clocks = len(clocks)

	for i, clock in enumerate(clocks):
		_plot_best_strat_dataset_clock_single(layout[f"best_strat_{data_cfg.key}_{clock}"],
			data=plot_data, data_cfg=data_cfg, clock=clock, series_cfgs=series_cfgs,
			dataset_cfg=dataset_cfgs[data_cfg.key], clock_cfg=clock_cfgs[clock],
			add_legend=(i == n_clocks - 1) and (data_cfg.key == "RB_SYF"),
			show_xticks=(i > n_clocks - 3), show_yticks=(i % 2 == 0),
			show_xlabel=(i == n_clocks - 1), show_ylabel=(i == 2),
			add_dataset_label=(i == n_clocks - 1),
		)

	return


def _plot_best_strat_dataset(layout: dict[str], data_cfgs: list[BestStratDataCfg],
	*, dataset_cfgs: pylib.DatasetCfgLib, clocks: list[str],
	series_cfgs: list[BestStratSeriesCfg],
	skip: bool = False,
) -> None:
	if skip:
		pylib.logger.info("skipping plotting best strat dataset")
		return

	for data_cfg in data_cfgs:
		_plot_best_strat_dataset_single(layout, data_cfg, clocks=clocks,
			series_cfgs=series_cfgs, dataset_cfgs=dataset_cfgs,
		)
	return


def _main():
	# configs
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	strat_sele_metric_cfgs = [
		StratSeleMetricCfg("mae", "MAE", unit="years"),
	]
	strat_sele_key_fields = StratSeleKeyFieldList([
		StratSeleKeyFieldList.KeyFieldCfg(key="depth_filtering", short_name="DF",
			full_name="Depth Filtering", color="#c4fbff",
			# get_active_label=lambda self, v: f"{self.short_name}$_{{{getattr(v, self.key)}}}$"
		),
		StratSeleKeyFieldList.KeyFieldCfg(key="imputation", short_name="IM",
			full_name="0/1 Imputation", color="#daffc3",
		),
		StratSeleKeyFieldList.KeyFieldCfg(key="transfer_learning", short_name="TL",
			full_name="Transfer Learning", color="#ecb3ff",
		),
	])
	downsample_series = ["mae", "perc_20x"]
	downsample_clocks = ["horvath2013", "skinandblood", "grimage",
		"dnamphenoage", "pchorvath2013", "zhangblup"]
	best_strat_data_cfgs = [
		BestStratDataCfg(key="RB_SYF",
			file="srrsh_filter_imput_transfer.age_pred.pkl"
		),
		BestStratDataCfg(key="BUCCAL_TWIST",
			file="BUCCAL_TWIST_filter_imput_transfer.age_pred.pkl"
		),
	]
	best_strat_series_cfgs = [
		BestStratSeriesCfg(key=(0, False, False), display_name="Direct",
			color_override="#c0c0c0", base_zorder=5, text_xy=(0.02, 0.98),
			text_props=dict(horizontalalignment="left", verticalalignment="top"),
		),
		BestStratSeriesCfg(key=(0, True, True), display_name="IM+TL",
			base_zorder=10, text_xy=(0.98, 0.02), marker="^",
			text_props=dict(horizontalalignment="right", verticalalignment="bottom"),
		),

	]
	best_strat_clocks = downsample_clocks

	# load data
	strat_sele_key_cfgs, strat_sele_clocks, strat_sele_results \
		= _strat_sele_load_results("srrsh_filter_imput_transfer.age_pred.pkl")
	dataset_cfgs = pylib.DatasetCfgLib.from_json()
	clock_cfgs = pylib.clock_cfg.ClockCfgLib.from_json()

	# prep data
	strat_sele_stats = _strat_sele_calc_stats(strat_sele_results,
		strat_sele_clocks)

	# plot
	layout = setup_layout(
		strat_sele_key_cfgs=strat_sele_key_cfgs,
		strat_sele_metric_cfgs=strat_sele_metric_cfgs,
		downsample_series=downsample_series,
		best_strat_data_cfgs=best_strat_data_cfgs,
		best_strat_clocks=best_strat_clocks,
	)

	figure = layout["figure"]

	_plot_downsample(layout, downsample_clocks,
		clock_cfgs=clock_cfgs,
		skip=False,
	)

	for i, metric in enumerate(strat_sele_metric_cfgs):
		key_order = _sort_stat_keys(strat_sele_stats, metric, reverse=False)

		_strat_sele_plot_stats(
			layout[f"strat_sele_box_{metric.key}"],
			layout[f"strat_sele_key_{metric.key}"],
			stats=strat_sele_stats,
			metric=metric,
			key_order=key_order,
			key_fields=strat_sele_key_fields,
			highlight_clock_cfgs=clock_cfgs,
			box_ec="#a0a0a0",
			box_fc="#e0e0e0",
		)

	_plot_best_strat_dataset(layout, best_strat_data_cfgs,
		series_cfgs=best_strat_series_cfgs,
		clocks=best_strat_clocks,
		dataset_cfgs=dataset_cfgs,
		skip=False,
	)

	figure.savefig("fig_6.svg", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
