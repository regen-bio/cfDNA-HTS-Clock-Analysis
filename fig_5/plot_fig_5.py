#!/usr/bin/env python3

import dataclasses
import json
import pdb
import pickle

import matplotlib
import matplotlib.axes
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.patches
import matplotlib.pyplot
import matplotlib.transforms
import mpllayout
import numpy
import sklearn.metrics
import pandas

import pylib


@dataclasses.dataclass
class MetricCfg(object):
	metric_key: str
	normalized: bool
	metric_display_name: str

	@property
	def norm_key(self) -> str:
		if self.normalized:
			ret = "std"
		else:
			ret = "raw"
		return ret

	@property
	def norm_display_name(self) -> str:
		if self.normalized:
			ret = "Standardized"
		else:
			ret = "Plain"
		return ret

	@property
	def full_display_name(self) -> str:
		if self.normalized:
			ret = f"{self.metric_display_name} (standardized)"
		else:
			ret = self.metric_display_name
		return ret


@dataclasses.dataclass
class HighlightPair(object):
	pair: tuple[str, str]
	color: str
	marker: str

	@property
	def key(self) -> str:
		return ("-").join(self.pair)

	def get_label(self, dataset_cfgs: pylib.DatasetCfgLib) -> str:
		return ("~").join([dataset_cfgs[d].display_name for d in self.pair])


@dataclasses.dataclass
class TransferMethodCfg(object):
	key: str
	display_name: str


def int_formatter(v) -> str:
	abs_v = abs(v)
	if abs_v >= 1e6:
		ret = f"{v / 1e6:.1f}M"
	elif abs_v >= 1e3:
		ret = f"{v / 1e3:.1f}K"
	else:
		ret = str(v)
	return ret


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


def _load_transf_results(fname: str, prefix: str = None) -> dict:
	with open(pylib.util.try_load_cache(fname), "rb") as fp:
		ret = pickle.load(fp)
	return ret


def _load_plain_results(datasets: list[str], prefix: str = None
) -> dict[str, pandas.DataFrame]:
	ret = dict()
	for ds in datasets:
		fname = pylib.util.try_load_cache(f"data/{ds}/age_pred.tsv")
		ret[ds] = pandas.read_csv(fname, sep="\t", index_col=0)
	return ret


def _extract_transf_results(full_results: dict, method: TransferMethodCfg,
) -> dict[str, pandas.DataFrame]:
	ret = dict()
	for ds, ds_res in full_results.items():
		ret[ds] = ds_res[method.key]["preds"]
	return ret


def _read_repl_group_json(datasets: list[str], prefix: str = None
) -> dict[str, list[dict]]:
	ret = dict()
	for ds in datasets:
		ret[ds] = pylib.dataset.ReplGroup.load_dataset(ds)
	return ret


def _prep_age_predict_stat_per_dataset(
	age_pred: pandas.DataFrame,
	repl_group: list[dict],
	clocks: list[str],
) -> dict[str, dict[str, numpy.ndarray]]:

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
) -> dict[str, dict[str, dict[str, numpy.ndarray]]]:
	ret = dict()
	for ds in age_preds.keys():
		ret[ds] = _prep_age_predict_stat_per_dataset(
			age_pred=age_preds[ds],
			repl_group=repl_groups[ds], clocks=clocks
		)
	return ret


def setup_layout(n_datasets: int, transf_comp_cfgs: list[CompCfg],
	residual_dist_metric_cfgs: list[MetricCfg],
) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.8,
		right_margin=0.2,
		top_margin=0.5,
		bottom_margin=1.6,
	)

	# transfer learning section
	n_transf_comp_cfgs = len(transf_comp_cfgs)
	transf_comp_w = n_datasets * 2.0
	transf_comp_h = 2.0
	transf_comp_gap_h = 1.2
	transf_comp_section_h = transf_comp_h * n_transf_comp_cfgs \
		+ transf_comp_gap_h * (n_transf_comp_cfgs - 1)
	for i, comp_cfg in enumerate(transf_comp_cfgs):
		transf_comp = lc.add_frame(f"transf_comp_{comp_cfg.key}")
		transf_comp.set_anchor("bottomleft", offsets=(
			0, (transf_comp_h + transf_comp_gap_h) * i))
		transf_comp.set_size(transf_comp_w, transf_comp_h)

	layout = lc.create_figure_layout()

	return layout


def _plot_transf_age_pred_comp(axes: matplotlib.axes.Axes, *,
	transf_stats: dict[str, dict[str, dict[str, numpy.ndarray]]],
	plain_stats: dict[str, dict[str, dict[str, numpy.ndarray]]],
	transf_method: TransferMethodCfg, datasets: list[str], clocks: list[str],
	dataset_cfgs: pylib.DatasetCfgLib, comp_cfg: CompCfg,
	add_legend: bool = True,
):
	n_datasets = len(datasets)
	n_clocks = len(clocks)

	dataset_w = 0.9
	clock_w = dataset_w / n_clocks
	clock_gap = clock_w * 0.15
	n_boxes = 3
	box_stretch = 1.2
	box_w = (clock_w - clock_gap) / n_boxes * box_stretch
	box_offset = (n_boxes / box_stretch - 1) / (n_boxes - 1) * box_w \
		if n_boxes > 1 else 0
	box_first_to_clock = -(clock_w - clock_gap - box_w) / 2

	for i_ds, ds in enumerate(datasets):
		for i_c, c in enumerate(clocks):
			clock_pos = i_ds + (1 - dataset_w) / 2 + (i_c + 0.5) * clock_w
			# positions
			teacher_pos = clock_pos + box_first_to_clock
			student_plain_pos = clock_pos + box_first_to_clock + box_offset * 1
			student_transf_pos = clock_pos + box_first_to_clock + box_offset * 2
			####################################################################
			# plot teacher
			edgecolor = "#a0a0a0"
			facecolor = "#c0c0c080"
			b = axes.boxplot(plain_stats[ds][c][comp_cfg.key],
				positions=[teacher_pos], widths=box_w,
				orientation="vertical", patch_artist=True,
				showmeans=False, showfliers=False,
				boxprops=dict(edgecolor=edgecolor, facecolor=facecolor,
					linewidth=0.5),
				medianprops=dict(color=edgecolor, linewidth=0.5),
				whiskerprops=dict(color=edgecolor, linewidth=0.5),
				capprops=dict(color=edgecolor, linewidth=0.5),
				capwidths=box_w * 0.8,
				zorder=5,
			)
			####################################################################
			# plot student, no transfer
			edgecolor = dataset_cfgs[ds].color
			facecolor = dataset_cfgs[ds].color + "30"
			b = axes.boxplot(transf_stats[ds]["no_transfer"][comp_cfg.key],
				positions=[student_plain_pos], widths=box_w,
				orientation="vertical", patch_artist=True,
				showmeans=False, showfliers=False,
				boxprops=dict(edgecolor=edgecolor, facecolor=facecolor,
					linewidth=0.5),
				medianprops=dict(color=edgecolor, linewidth=0.5),
				whiskerprops=dict(color=edgecolor, linewidth=0.5),
				capprops=dict(color=edgecolor, linewidth=0.5),
				capwidths=box_w * 0.8,
				zorder=10,
			)
			# add a white patch to cover the low-level boxes
			for b in b["boxes"]:
				p = matplotlib.patches.PathPatch(b.get_path(),
					facecolor="#ffffff", edgecolor="none",
					zorder=9,
				)
				axes.add_patch(p)
			####################################################################
			# plot student, transferred
			edgecolor = "#000000"
			facecolor = dataset_cfgs[ds].color + "80"
			b = axes.boxplot(transf_stats[ds][c]["true_ad"],
				positions=[student_transf_pos], widths=box_w,
				orientation="vertical", patch_artist=True,
				showmeans=False, showfliers=False,
				boxprops=dict(edgecolor=edgecolor, facecolor=facecolor,
					linewidth=0.5),
				medianprops=dict(color=edgecolor, linewidth=0.5),
				whiskerprops=dict(color=edgecolor, linewidth=0.5),
				capprops=dict(color=edgecolor, linewidth=0.5),
				capwidths=box_w * 0.8,
				zorder=15,
			)
			# add a white patch to cover the low-level boxes
			for b in b["boxes"]:
				p = matplotlib.patches.PathPatch(b.get_path(),
					facecolor="#ffffff", edgecolor="none",
					zorder=14,
				)
				axes.add_patch(p)

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
				matplotlib.patches.Patch(linewidth=0.5, edgecolor="#a0a0a0",
					facecolor="#c0c0c080",
				)
			],
			label="Teacher model"
		)
		handles.append(p)
		p = MultiPatchDummy(
			patches=[
				matplotlib.patches.Patch(linewidth=0.5,
					edgecolor=dataset_cfgs[ds].color,
					facecolor=dataset_cfgs[ds].color + "30",
				)
				for ds in datasets
			],
			label=f"Independently trained"
		)
		handles.append(p)
		p = MultiPatchDummy(
			patches=[
				matplotlib.patches.Patch(linewidth=0.5, edgecolor="#000000",
					facecolor=dataset_cfgs[ds].color + "80",
				)
				for ds in datasets
			],
			label=transf_method.display_name,
		)
		handles.append(p)
		handler_map = matplotlib.legend.Legend.get_default_handler_map().copy()
		handler_map[MultiPatchDummy] = MultiPatchHandler()
		legend = axes.legend(handles=handles, handler_map=handler_map, loc=9,
			bbox_to_anchor=[0.5, -0.5], frameon=True, handlelength=3.5,
			ncol=len(handles),
		)

	# misc
	for x in range(1, n_datasets):
		axes.axvline(x, color="#808080", linewidth=0.5)
	axes.set_xlim(0, n_datasets)

	xticks = list()
	for di in range(n_datasets):
		for ci in range(n_clocks):
			xticks.append(di + 0.5 - dataset_w / 2 + (ci + 0.5) * clock_w)
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


def _main():
	# configs
	box_metric_cfgs = [
		MetricCfg(metric_key="mmd", normalized=False,
			metric_display_name="MMD"),
		MetricCfg(metric_key="mmd", normalized=True,
			metric_display_name="MMD"),
		MetricCfg(metric_key="wasserstein", normalized=False,
			metric_display_name="Wasserstein"),
		MetricCfg(metric_key="wasserstein", normalized=True,
			metric_display_name="Wasserstein"),
	]
	box_highlight_pairs = [
		HighlightPair(pair=("RB_935K", "RB_MSA"),
			color="#404040", marker="s"),
		HighlightPair(pair=("RB_935K", "RB_GDNA_GALAXY"),
			color="#88AE4F", marker="H"),
		HighlightPair(pair=("RB_GALAXY", "RB_TWIST"),
			color="#0001a1", marker="^"),
		HighlightPair(pair=("RB_GALAXY", "RB_GDNA_GALAXY"),
			color="#037f77", marker="D"),
		HighlightPair(pair=("RB_GDNA_GALAXY", "RB_GDNA_TWIST"),
			color="#c5272d", marker="o"),
		HighlightPair(pair=("RB_TWIST", "RB_GDNA_TWIST"),
			color="#da8f45", marker="v"),
	]
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	clocks = ["horvath2013", "skinandblood", "grimage", "dnamphenoage",
		"pchorvath2013", "zhangblup"]
	transf_method = TransferMethodCfg(key="distill",
		display_name="Distillation")
	comp_cfgs = [
		CompCfg(key="repl_ad", display_name="RD"),
		CompCfg(key="true_ad", display_name="MAE"),
	]

	# load data
	n_datasets = len(datasets)
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# plot
	layout = setup_layout(n_datasets,
		transf_comp_cfgs=comp_cfgs,
		residual_dist_metric_cfgs=box_metric_cfgs,
	)
	figure = layout["figure"]

	_plot_transf_section(layout,
		datasets=datasets,
		clocks=clocks,
		transf_method=transf_method,
		comp_cfgs=comp_cfgs,
		dataset_cfgs=dataset_cfgs
	)

	figure.savefig("fig_5.svg", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
