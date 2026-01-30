#!/usr/bin/env python3

import dataclasses
import pdb
import pickle

import matplotlib
import matplotlib.pyplot
import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.lines
import matplotlib.patches
import matplotlib.transforms
import mpllayout
import numpy
import pandas
import sklearn
import sklearn.metrics

import pylib


@dataclasses.dataclass
class MetricCfg(object):
	key: str
	normalized: bool
	display_name: str


@dataclasses.dataclass
class TransferMethodCfg(object):
	key: str
	display_name: str


@dataclasses.dataclass
class CompCfg(object):
	key: str
	display_name: str
	unit: str = "years"

	@property
	def ylabel(self) -> str:
		return f"{self.display_name} ({self.unit})"


class MultiPatchDummy(matplotlib.patches.Patch):
	def __init__(self, *ka, patches: list[matplotlib.patches.Patch], **kw):
		self.patches = patches
		return super().__init__(*ka, **kw)


class MultiPatchHandler(matplotlib.legend_handler.HandlerPatch):
	def create_artists(self, legend: matplotlib.legend.Legend,
		orig_handle: MultiPatchDummy, xdescent: float, ydescent: float,
		width: float, height: float, fontsize: float,
		transform: matplotlib.transforms.Transform
	) -> list:
		patches = orig_handle.patches
		n_patches = len(patches)

		gap_frac = 0.20
		patch_w = width / (1 * n_patches + gap_frac * (n_patches - 1))
		gap_w = patch_w * gap_frac

		artists = list()
		for i, patch in enumerate(patches):
			x = (patch_w + gap_w) * i
			p = matplotlib.patches.Rectangle((x, 0),
				patch_w, height,
				linewidth=patch.get_linewidth(),
				edgecolor=patch.get_edgecolor(),
				facecolor=patch.get_facecolor(),
			)
			artists.append(p)
			l = matplotlib.lines.Line2D([x, x + patch_w], [0, height],
				color=patch.get_edgecolor(),
				linewidth=patch.get_linewidth(),
			)
			artists.append(l)

		return artists


def setup_layout(datasets: list[str], comp_cfgs: list[CompCfg]) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.8,
		right_margin=0.2,
		top_margin=0.5,
		bottom_margin=1.8,
	)

	# cross test section
	n_datasets = len(datasets)

	cross_offset_w = 0.8
	cross_row_h = 0.8
	cross_w = 1.5
	cross_h = cross_row_h * n_datasets
	cross_gap_w = 0.1

	for i, dataset in enumerate(datasets):
		cross = lc.add_frame(f"cross_{dataset}")
		cross.set_anchor("bottomleft",
			offsets=((cross_w + cross_gap_w) * i + cross_offset_w, 0),
		)
		cross.set_size(cross_w, cross_h)

	# naive transfer section
	transf_cross_gap_h = 1.6
	transf_cross_hang_r = (1.6 - cross_w) * n_datasets
	transf_comp_w = cross_w * n_datasets + cross_gap_w * (n_datasets - 1) \
		+ cross_offset_w + transf_cross_hang_r
	transf_comp_h = 1.8
	transf_comp_gap_h = 1.2
	transf_offset_h = cross_h + transf_cross_gap_h
	for i, comp_cfg in enumerate(comp_cfgs):
		transf_comp = lc.add_frame(f"transf_comp_{comp_cfg.key}")
		transf_comp.set_anchor("bottomleft",
			offsets=(0, (transf_comp_h + transf_comp_gap_h) * i + transf_offset_h)
		)
		transf_comp.set_size(transf_comp_w, transf_comp_h)

	layout = lc.create_figure_layout()

	return layout


def _load_transfer_results(train: str, test: str) -> dict[str]:
	fname = f"cross_test.{train}.{test}.no_imput.age_pred.pkl"
	with open(fname, "rb") as fp:
		ret = pickle.load(fp)
	return ret


def _add_clock_scatter(axes: matplotlib.axes.Axes, values: pandas.Series, pos: float,
	*, clocks: list[str], clock_cfgs: pylib.ClockCfgLib,
	highlight_clocks: set[str] | None = None,
) -> dict[str]:
	if highlight_clocks is None:
		highlight_clocks = set(clocks)

	handles = dict()
	for c in clocks:
		clock_cfg: pylib.ClockCfg = clock_cfgs[c]
		if c in highlight_clocks:
			marker = clock_cfg.marker
			size = 30
			edgecolor = "#000000"
			facecolor = clock_cfg.color + "a0"
			zorder = 20
			label = clock_cfg.key
		else:
			marker = "o"
			size = 12
			edgecolor = "#a0a0a0"
			facecolor = "#a0a0a0"
			zorder = 15
			label = "[other clocks]"

		p = axes.scatter(values[c], pos, marker=marker, s=size, linewidth=0.5,
			edgecolors=edgecolor, facecolors=facecolor, zorder=20, label=label,
		)
		handles[label] = p

	return handles


def _read_repl_group_json(datasets: list[str], prefix: str = None
) -> dict[str, list[dict]]:
	ret = dict()
	for ds in datasets:
		ret[ds] = pylib.dataset.ReplGroup.load_dataset(ds)
	return ret


def _load_transf_results(fname: str, prefix: str = None) -> dict:
	with open(pylib.util.try_load_cache(fname), "rb") as fp:
		ret = pickle.load(fp)
	return ret


def _extract_transf_results(full_results: dict, method: TransferMethodCfg,
) -> dict[str, pandas.DataFrame]:
	ret = dict()
	for ds, ds_res in full_results.items():
		ret[ds] = ds_res[method.key]["preds"]
	return ret


def _load_plain_results(datasets: list[str], prefix: str = None
) -> dict[str, pandas.DataFrame]:
	ret = dict()
	for ds in datasets:
		fname = pylib.util.try_load_cache(f"{ds}/final.filtered.age_pred.tsv")
		ret[ds] = pandas.read_csv(fname, sep="\t", index_col=0)
	return ret


def _prep_age_predict_stat_per_dataset(
	age_pred: pandas.DataFrame,
	repl_group: list[dict],
	clocks: list[str],
) -> dict[str, dict[str]]:

	# repl_to_subj = {r: v["subject_id"] for v in repl_group
	# for r in v["replicates"]}
	true_age = age_pred["age"].to_dict()

	ret = dict()

	for c in clocks:
		true_ad = list()
		true_std = list()
		repl_ad = list()
		repl_std = list()

		for g in repl_group:
			_pred = age_pred.loc[g["replicates"], c].values
			_true = age_pred.loc[g["replicates"], "age"].values

			# true ad/std
			true_ad.append(sklearn.metrics.mean_absolute_error(_true, _pred))
			true_std.append(sklearn.metrics.root_mean_squared_error(_true, _pred))

			# repl ad/std
			_mean = _pred.mean()
			repl_ad.append(numpy.abs(_pred - _mean).mean())
			repl_std.append(_pred.std(ddof=1))

		ret[c] = {
			"mae": numpy.mean(true_ad),
			"true_ad": numpy.asarray(true_ad, dtype=float),
			"true_std": numpy.asarray(true_std, dtype=float),
			"repl_ad": numpy.asarray(repl_ad, dtype=float),
			"repl_std": numpy.asarray(repl_std, dtype=float),
		}

	return ret


def _prep_age_predict_stat(
	age_preds: dict[str, pandas.DataFrame],
	repl_groups: dict[str, list[dict]],
	clocks: list[str],
) -> dict[str, dict[str, dict[str]]]:
	ret = dict()
	for ds in age_preds.keys():
		ret[ds] = _prep_age_predict_stat_per_dataset(
			age_pred=age_preds[ds],
			repl_group=repl_groups[ds], clocks=clocks
		)
	return ret


def _plot_transf_age_pred_series_single(axes: matplotlib.axes.Axes, *,
	vals: numpy.ndarray | float, pos: float, offset: float,
	edgecolor: str = "#000000", facecolor: str = "none", zorder_ref: int = 0,
	b_width: float = 0.5, s_size: float = 40,
	label: str = None,
) -> matplotlib.artist.Artist:
	if isinstance(vals, numpy.ndarray):
		b = axes.boxplot(vals,
			positions=[pos + offset], widths=b_width,
			orientation="vertical", patch_artist=True,
			showmeans=False, showfliers=False,
			boxprops=dict(edgecolor=edgecolor, facecolor=facecolor,
				linewidth=0.5),
			medianprops=dict(color=edgecolor, linewidth=0.5),
			whiskerprops=dict(color=edgecolor, linewidth=0.5),
			capprops=dict(color=edgecolor, linewidth=0.5),
			capwidths=b_width * 0.8,
			zorder=zorder_ref + 1,
		)
		for b in b["boxes"]:
			p = matplotlib.patches.PathPatch(b.get_path(),
				facecolor="#ffffff", edgecolor="none",
				zorder=zorder_ref,
			)
			axes.add_patch(p)
		# to return as handle
		ret = matplotlib.patches.Patch(linewidth=0.5, edgecolor="#ffffff",
			facecolor="#ffffff", label=label,
		)
	else:
		ret = axes.scatter([pos], [vals], marker="o", s=40,
			linewidth=0.5, edgecolors=edgecolor, facecolors=facecolor,
			zorder=zorder_ref, label=label,
		)

	return ret


def _plot_transf_age_pred_comp(axes: matplotlib.axes.Axes, *,
	transf_stats: dict[str, dict[str, dict[str, numpy.ndarray]]],
	plain_stats: dict[str, dict[str, dict[str, numpy.ndarray]]],
	transf_method: TransferMethodCfg, datasets: list[str], clocks: list[str],
	dataset_cfgs: pylib.DatasetCfgLib, comp_cfg: CompCfg,
	add_legend: bool = True,
):
	n_datasets = len(datasets)
	n_clocks = len(clocks)

	clock_ref = (numpy.arange(0.5, n_clocks, step=1) /
	    n_clocks * 0.9 + 0.05) * n_clocks
	teacher_offset = -0.15
	transf_offset = 0.15
	box_width = 0.4
	scatter_size = 40

	for i_ds, ds in enumerate(datasets):
		for i_c, c in enumerate(clocks):
			pos = clock_ref[i_c] + n_clocks * i_ds

			# teacher
			edgecolor = "#a0a0a0"
			facecolor = "#c0c0c080"
			vals = plain_stats[ds][c][comp_cfg.key]
			h = _plot_transf_age_pred_series_single(axes, vals=vals,
				pos=pos, offset=teacher_offset,
				edgecolor=edgecolor, facecolor=facecolor,
				zorder_ref=10, b_width=box_width, s_size=scatter_size,
				label="Direct (teacher)",
			)

			# TL
			dataset_cfg = dataset_cfgs[ds]
			edgecolor = dataset_cfg.color
			facecolor = dataset_cfg.color + "40"
			vals = transf_stats[ds][c][comp_cfg.key]
			h = _plot_transf_age_pred_series_single(axes, vals=vals,
				pos=pos, offset=transf_offset,
				edgecolor=edgecolor, facecolor=facecolor,
				zorder_ref=15, b_width=box_width, s_size=scatter_size,
				label=transf_method.display_name,
			)

		# add dataset text
		axes.text((i_ds + 0.5) / n_datasets, 1.03,
			dataset_cfgs[ds].display_name,
			transform=axes.transAxes,
			horizontalalignment="center", verticalalignment="bottom",
		)

	# legend
	if add_legend:
		handles = list()
		p = MultiPatchDummy(
			patches=[
				matplotlib.patches.Patch(
					linewidth=0.5,
					edgecolor="#a0a0a0",
					facecolor="#c0c0c080",
				)
			],
			label="Direct (teacher)"
		)
		handles.append(p)
		p = MultiPatchDummy(
			patches=[
				matplotlib.patches.Patch(
					linewidth=0.5,
					edgecolor=(color := dataset_cfgs[ds].color),
					facecolor=color + "40",
				)
				for ds in datasets
			],
			label=transf_method.display_name,
		)
		handles.append(p)
		handler_map = matplotlib.legend.Legend.get_default_handler_map().copy()
		handler_map[MultiPatchDummy] = MultiPatchHandler()
		legend = axes.legend(handles=handles, handler_map=handler_map, loc=9,
			bbox_to_anchor=[0.5, -0.5], frameon=False, handlelength=3.5,
			ncol=len(handles),
		)

	# misc
	for x in range(1, n_datasets):
		axes.axvline(x * n_clocks, color="#808080", linewidth=0.5)
	axes.set_xlim(0, n_datasets * n_clocks)

	xticks = list()
	for di in range(n_datasets):
		xticks.extend(clock_ref + n_clocks * di)
	axes.set_xticks(xticks)
	axes.set_xticklabels(clocks * n_datasets, rotation=40,
		horizontalalignment="right", verticalalignment="top",
	)
	axes.set_ylabel(comp_cfg.ylabel)

	return


def _plot_transf_section(layout: dict, *, datasets: list[str], clocks: list[str],
	transf_method: TransferMethodCfg, comp_cfgs: list[CompCfg],
	dataset_cfgs: pylib.DatasetCfgLib,
) -> None:
	# load data
	repl_groups = _read_repl_group_json(datasets)
	transf_preds_full = _load_transf_results(
		"transfer_learn.age_pred.pkl")
	transf_preds = _extract_transf_results(transf_preds_full,
		method=transf_method)
	plain_preds = _load_plain_results(datasets)

	transf_stats = _prep_age_predict_stat(transf_preds, repl_groups,
		["no_transfer"] + clocks)
	plain_stats = _prep_age_predict_stat(plain_preds, repl_groups, clocks)

	for comp_cfg in comp_cfgs:
		_plot_transf_age_pred_comp(layout[f"transf_comp_{comp_cfg.key}"],
			transf_stats=transf_stats,
			plain_stats=plain_stats,
			transf_method=transf_method,
			datasets=datasets,
			clocks=clocks,
			dataset_cfgs=dataset_cfgs,
			comp_cfg=comp_cfg,
			add_legend=(comp_cfg == comp_cfgs[0]),
		)
	return


def _plot_cross_transf_single(axes: matplotlib.axes.Axes, *,
	datasets: list[str], test_dataset: str,
	clocks: list[str],
	clock_cfgs: pylib.ClockCfgLib,
	metric_cfg: MetricCfg,
	dataset_cfgs: pylib.DatasetCfgLib,
	highlight_clocks: list[str] = None,
	show_yticklabel: bool = False,
	show_rticklabel: bool = False,
) -> None:

	# load data
	n_datasets = len(datasets)
	repl_group = pylib.dataset.ReplGroup.load_dataset(test_dataset)
	results = {train_dataset: _load_transfer_results(train_dataset, test_dataset)
	    for train_dataset in datasets if train_dataset != test_dataset}

	# calculate stats
	base_pred = None
	base_stats = None

	cross_stats = dict()
	for train_dataset in datasets:
		if train_dataset not in results:
			continue
		cross_pred = results[train_dataset]["cross_pred"]
		stat = pylib.AgePredRes(cross_pred).calc_stat(clocks, repl_group=repl_group)
		cross_stats[train_dataset] = stat
		#
		if base_pred is None:
			base_pred = results[train_dataset]["base_pred"]
			base_stats = pylib.AgePredRes(base_pred).calc_stat(
				clocks, repl_group=repl_group)

	# plot
	test_dataset_cfg = dataset_cfgs[test_dataset]
	clock_handles = dict()
	width = 0.35
	capwidth = width * 0.8

	# per cross-test dataset
	for i, train_dataset in enumerate(datasets):

		# base stat, without transfer cross-test
		pos = n_datasets - i - 0.7
		if train_dataset in cross_stats:
			vals = base_stats[metric_cfg.key]
			p = axes.boxplot(vals, positions=[pos], orientation="horizontal",
				widths=width, showfliers=False, patch_artist=True,
				boxprops=dict(linewidth=0.5, edgecolor="#808080", facecolor="#e0e0e0"),
				medianprops=dict(linewidth=1.0, color="#808080"),
				whiskerprops=dict(linewidth=0.5, color="#808080"),
				capprops=dict(linewidth=0.5, color="#808080"),
				capwidths=capwidth, zorder=10,
			)
			h = _add_clock_scatter(axes, vals, pos,
				clocks=clocks,
				clock_cfgs=clock_cfgs,
				highlight_clocks=highlight_clocks,
			)
			clock_handles.update(h)
		if show_rticklabel:
			axes.text(1.05, pos / n_datasets, "Direct", fontsize=8, color="#606060",
			 	horizontalalignment="left", verticalalignment="center",
				transform=axes.transAxes,
			)

		# cross stat
		pos = n_datasets - i - 0.3
		train_dataset_cfg = dataset_cfgs[train_dataset]
		if train_dataset in cross_stats:
			vals = cross_stats[train_dataset][metric_cfg.key]
			p = axes.boxplot(vals, positions=[pos], orientation="horizontal",
				widths=width, showfliers=False, patch_artist=True,
				boxprops=dict(linewidth=0.5, edgecolor=train_dataset_cfg.color,
					facecolor=train_dataset_cfg.color + "40"),
				medianprops=dict(linewidth=1.0, color=train_dataset_cfg.color),
				whiskerprops=dict(linewidth=0.5, color=train_dataset_cfg.color),
				capprops=dict(linewidth=0.5, color=train_dataset_cfg.color),
				capwidths=capwidth, zorder=10,
			)
			h = _add_clock_scatter(axes, vals, pos,
				clocks=clocks,
				clock_cfgs=clock_cfgs,
				highlight_clocks=highlight_clocks,
			)
			clock_handles.update(h)
		if show_rticklabel:
			axes.text(1.05, pos / n_datasets, "TL", fontsize=8, color=train_dataset_cfg.color,
			 	horizontalalignment="left", verticalalignment="center",
				transform=axes.transAxes,
			)

		p = matplotlib.patches.Rectangle((0, i + 0.5), 100, 0.5,
			edgecolor="none", facecolor="#f0f0f0", zorder=0,
		)
		axes.add_artist(p)

	# clock legend
	handles = sorted(clock_handles.values(), key=lambda h: h.get_label())
	legend = axes.figure.legend(handles=handles, loc=8, bbox_to_anchor=[0.5, 0.02],
		ncols=5, handlelength=0.75, fontsize=8, title="Clock markers",
		title_fontsize=10, frameon=False,
	)

	# grid
	axes.grid(axis="x", linestyle="-", linewidth=0.5, color="#e0e0e0")
	for y in range(1, n_datasets + 1):
		axes.axhline(y, linestyle="-", linewidth=0.5, color="#000000")

	# misc
	for sp in axes.spines.values():
		sp.set(linewidth=1.0)
	axes.tick_params(
		left=True, labelleft=show_yticklabel,
		right=False, labelright=False,
		top=False, labeltop=False,
		bottom=True, labelbottom=True,
		# color=test_dataset_cfg.color,
	)
	# axes.set_facecolor(test_dataset_cfg.color + "0c")
	axes.set_xlim(0, None)
	axes.set_ylim(0, n_datasets)
	axes.set_xlabel(metric_cfg.display_name, fontsize=8)
	for l in axes.xaxis.get_ticklabels():
		l.set(fontsize=8)
	axes.set_yticks(numpy.arange(n_datasets) + 0.5)
	axes.set_yticklabels(
		[f"Training: {dataset_cfgs[d].display_name}" for d in datasets[::-1]],
		fontsize=8,
	)
	for l, dataset in zip(axes.yaxis.get_ticklabels(), datasets[::-1]):
		l.set(color=dataset_cfgs[dataset].color)
	title = f"Testing: {test_dataset_cfg.display_name}"
	axes.set_title(title, fontsize=9, color=test_dataset_cfg.color)

	return


def _main():
	# configs
	comp_cfgs = [
		CompCfg(key="repl_ad", display_name="RD"),
		CompCfg(key="mae", display_name="MAE"),
	]
	metric_cfg = MetricCfg(
		key="mae",
		normalized=False,
		display_name="MAE (years)",
	)
	datasets = [
		"RB_GALAXY",
		"RB_TWIST",
		"RB_GDNA_GALAXY",
		"RB_GDNA_TWIST",
	]
	clocks = pylib.clock.load_clocks(exclude={"altumage"})
	clock_cfgs = pylib.ClockCfgLib.from_json()
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# plot
	layout = setup_layout(datasets, comp_cfgs)
	figure: matplotlib.figure.Figure = layout["figure"]

	# naive transfer section
	_plot_transf_section(layout,
		datasets=["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"],
		clocks=["horvath2013", "skinandblood", "grimage", "dnamphenoage",
			"pchorvath2013", "zhangblup"],
		transf_method=TransferMethodCfg(key="distill",
			display_name="TL"),
		comp_cfgs=comp_cfgs,
		dataset_cfgs=dataset_cfgs
	)

	# align axes
	ref_axes = layout[f"cross_{datasets[0]}"]
	for dataset in datasets[1:]:
		axes: matplotlib.axes.Axes = layout[f"cross_{dataset}"]
		axes.sharex(ref_axes)

	# cross-test
	for i, test_dataset in enumerate(datasets):
		axes: matplotlib.axes.Axes = layout[f"cross_{test_dataset}"]
		axes.sharex(ref_axes)

		_plot_cross_transf_single(axes,
			datasets=datasets,
			test_dataset=test_dataset,
			clocks=clocks,
			clock_cfgs=clock_cfgs,
			metric_cfg=metric_cfg,
			dataset_cfgs=dataset_cfgs,
			highlight_clocks=None,
			show_yticklabel=(i == 0),
			show_rticklabel=(i == len(datasets) - 1),
		)

	figure.savefig("fig_5.png", dpi=600)
	figure.savefig("fig_5.svg", dpi=600)
	figure.savefig("fig_5.pdf", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
