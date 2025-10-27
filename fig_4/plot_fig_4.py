#!/usr/bin/env python3

import collections
import dataclasses
import itertools
import json
import pdb
import pickle

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot
import matplotlib.ticker
import mpllayout
import numpy
import sklearn.metrics
import threadpoolctl
import tqdm
import pandas

import pylib


@dataclasses.dataclass
class ImputImpactCfg(object):
	key: str
	display_name: str
	unit: str
	add_legend: bool = False

	@property
	def ylabel(self) -> str:
		return f"{self.display_name} ({self.unit})"


@dataclasses.dataclass
class ImputCfg(object):
	key: str
	display_name: str


def load_ill_cpg_surv_data(fname: str) -> dict[str, dict]:
	with open(fname, "r") as fp:
		ret = json.load(fp)
	# convert ill_depth_count keys from str to int
	for v in ret.values():
		v["ill_depth_count"] = {int(k): val
			for k, val in v["ill_depth_count"].items()}
	return ret


def _prep_age_predict_stat(
	age_pred: pandas.DataFrame,
	repl_group: pylib.dataset.ReplGroup,
	clocks: list[str],
) -> dict[str, numpy.ndarray]:
	mae_meth = sklearn.metrics.mean_absolute_error
	rmse_meth = sklearn.metrics.root_mean_squared_error

	repl_to_subj = {r: v["subject_id"]
		for v in repl_group for r in v["replicates"]}

	true_age = {repl_to_subj[r]: age_pred.loc[r, "age"] for r in age_pred.index}
	# pred_age = age_pred[clocks].values

	# group replicate bys repl_group
	mae = list()
	rmse = list()
	repl_std = list()
	repl_mad = list()
	for g in repl_group:
		subj_pred_age = age_pred.loc[g["replicates"], clocks].values
		_mean = subj_pred_age.mean(axis=0)
		_std = subj_pred_age.std(axis=0, ddof=1)
		assert len(_mean) == len(_std) == len(clocks) == subj_pred_age.shape[1]
		_true = [true_age[g["subject_id"]]] * len(_mean)
		mae.append(mae_meth(_true, _mean))
		rmse.append(rmse_meth(_true, _mean))
		repl_std.append(numpy.mean(_std))
		repl_mad.append(
			numpy.abs(subj_pred_age - _mean.reshape(1, -1)).sum(axis=0).mean()
		)

	ret = {
		"mae": numpy.asarray(mae, dtype=float),
		"rmse": numpy.asarray(rmse, dtype=float),
		"repl_std": numpy.asarray(repl_std, dtype=float),
		"repl_mad": numpy.asarray(repl_mad, dtype=float),
	}
	return ret


def prep_full_imput_age_predict_stat(datasets: list[str], clocks: list[str]
) -> dict[str, dict]:
	# key order: dataset -> "plain" | "imput"
	# plain:  dict[str, ndarray]: metric
	# imput: dict[tuple[str, int], dict]: (imput, depth) -> metric
	ret = dict()

	for ds in datasets:
		ds_val = dict()

		# prep data
		repl_group = pylib.dataset.ReplGroup.load_dataset(ds)

		# read the pred without imputation
		fname = f"data/{ds}/age_pred.tsv"
		ds_val["plain"] = _prep_age_predict_stat(
			pandas.read_csv(fname, sep="\t", index_col=0), repl_group, clocks,
		)

		# load the pred with imputation
		with open(f"age_predict_with_imput_depth/{ds}.pkl", "rb") as fp:
			imput_res = pickle.load(fp)
		ds_val["imput"] = {k: _prep_age_predict_stat(v, repl_group, clocks)
			for k, v in imput_res.items()
		}

		ret[ds] = ds_val

	return ret


def extract_imput_age_predict_stat(full_stat: dict, depth: int = 0) -> dict:
	# key order = dataset -> imput
	ret = dict()

	for ds, full_v in full_stat.items():
		ds_res = dict()

		ds_res["plain"] = full_v["plain"]

		for k, v in full_v["imput"].items():
			if k[1] != depth:
				continue
			ds_res[k[0]] = v

		ret[ds] = ds_res

	return ret


def _full_sig_test_bh(full_stat: dict, conf_level: float = 5e-2
) -> dict[str, dict[tuple[str, int], dict[str, float]]]:
	pval_key = list()
	pval_1d = list()
	for ds, ds_v in full_stat.items():
		for (imput, depth), imput_v in ds_v["imput"].items():
			for metric, vals in imput_v.items():
				pval_key.append((ds, imput, depth, metric))
				pval_1d.append(_permutation_test(vals, ds_v["plain"][metric],
					alternative="two-sided")[0])
	pval_1d = numpy.asarray(pval_1d, dtype=float)
	n_vals = len(pval_1d)

	# b-h procedure
	q = (numpy.arange(n_vals) + 1) / n_vals * conf_level
	sort_idx = numpy.argsort(pval_1d)
	sig_flag = numpy.zeros(n_vals, dtype=bool)
	sig_flag[sort_idx] = pval_1d[sort_idx] <= q

	ret = collections.defaultdict(lambda: collections.defaultdict(dict))
	for flag, key, pval in zip(sig_flag, pval_key, pval_1d):
		ds, imput, depth, metric = key
		ret[ds][(imput, depth)][metric] = pval if flag else numpy.nan

	return ret


def setup_layout(ill_surv_datasets: list[str],
	imput_impact_datasets: list[str],
	imput_impact_cfgs: list[dict[ImputImpactCfg]],
) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=0.8,
		right_margin=0.2,
		top_margin=0.2,
		bottom_margin=0.7,
	)

	imput_impact_w = 2.6
	imput_impact_h = 1.2
	imput_impact_gap_w = 0.6
	imput_impact_gap_h = 0.0
	imput_top_gap_h = 0.8

	imput_depth_w = 2.8
	imput_depth_h = 1.0
	imput_depth_gap_h = 1.2
	imput_depth_offset_w = 0.8

	clock_cov_w = 2.6
	clock_cov_h = 0.5 * len(ill_surv_datasets)
	clock_cov_top_gap_h = 1.2
	clock_cov_extra_offset_h = 0.5

	# imput impact
	for ir, ds in enumerate(imput_impact_datasets[::-1]):  # top-down order
		for ic, imput in enumerate(imput_impact_cfgs):
			imput_impact = lc.add_frame(f"imput_impact_{ds}_{imput.key}")
			imput_impact.set_anchor("bottomleft", offsets=(
				(imput_impact_w + imput_impact_gap_w) * ic,
				(imput_impact_h + imput_impact_gap_h) * ir
			))
			imput_impact.set_size(imput_impact_w, imput_impact_h)

	imput_impact_topleft = lc.get_frame(
		f"imput_impact_{imput_impact_datasets[0]}_{imput_impact_cfgs[0].key}"
	)
	imput_impact_topright = lc.get_frame(
		f"imput_impact_{imput_impact_datasets[0]}_{imput_impact_cfgs[1].key}"
	)

	# imput vs depth
	imput_depth_reprod = lc.add_frame("imput_depth_reprod")
	imput_depth_reprod.set_anchor("bottomleft", imput_impact_topright,
		"topleft", offsets=(imput_depth_offset_w, imput_depth_gap_h))
	imput_depth_reprod.set_size(imput_depth_w, imput_depth_h)

	imput_depth_mae = lc.add_frame("imput_depth_mae")
	imput_depth_mae.set_anchor("bottomleft", imput_depth_reprod,
		"topleft", offsets=(0, 0.1))
	imput_depth_mae.set_size(imput_depth_w, imput_depth_h)

	# ill cpg clock coverate
	clock_cov = lc.add_frame("clock_cov_ill")
	clock_cov.set_anchor("bottomleft",
		ref_frame=imput_impact_topleft,
		ref_anchor="topleft",
		offsets=(0, imput_top_gap_h + clock_cov_extra_offset_h)
	)
	clock_cov.set_size(clock_cov_w, clock_cov_h)

	layout = lc.create_figure_layout()

	return layout


def _plot_clock_affected_cpg_frac(axes: matplotlib.axes.Axes, data: dict[str, dict], *,
	clocks: list[str], datasets: list[str] = None,
	dataset_cfgs: pylib.DatasetCfgLib, box_color: str = "#c0c0c0",
):
	n_clocks = len(clocks)

	if datasets is None:
		datasets = sorted(data.keys())

	for i, ds in enumerate(datasets):
		values = [data[ds]["clock_affected_frac"][c] for c in clocks]
		# box plot
		axes.boxplot(values, positions=[i], vert=False, widths=0.6,
			showmeans=False, medianprops=dict(linewidth=1.5, color=box_color),
			showcaps=True, capprops=dict(linewidth=1.0, color=box_color),
			showfliers=False, capwidths=0.6,
			patch_artist=True, boxprops=dict(linewidth=1.0, edgecolor=box_color,
				facecolor=box_color + "40"),
			whiskerprops=dict(linewidth=1.0, color=box_color),
			zorder=5,
		)
		# scatter
		axes.scatter(values, [i] * len(values), marker="o", s=40, linewidths=0.5,
			# edgecolors=dataset_cfgs[ds].color,
			edgecolors="#000000", facecolors=dataset_cfgs[ds].color + "80",
			zorder=8,
		)

	# add an invisible line at x=0 to force showing the y-axis
	p, *_ = axes.plot([0, 0], [0, 0], color="none")
	p.set_visible(False)

	# misc
	axes.axvline(0, linewidth=0.5, color="#c0c0c0")
	axes.set_ylim(len(datasets) - 0.5, -0.5)  # inversed y axis
	axes.set_yticks(numpy.arange(len(datasets)))
	yticklabels = [dataset_cfgs[ds].display_name.replace("_", "\n")
		for ds in datasets]
	axes.set_yticklabels(yticklabels)
	axes.set_xlabel(
		f"CpG% with unreliable beta\n(in {n_clocks} HCRL clocks)")
	axes.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

	return


def _plot_ill_cpg_survey(layout: dict, datasets: list[str], clocks: list[str],
	dataset_cfgs: pylib.DatasetCfgLib,
):
	# load beta and depth
	data = load_ill_cpg_surv_data("ill_cpg_stat.json")

	_plot_clock_affected_cpg_frac(layout["clock_cov_ill"], data,
		clocks=clocks, dataset_cfgs=dataset_cfgs, datasets=datasets,
		box_color="#c0c0c0",
	)
	return


def _permutation_test(x, y, n_permutations: int = 10000, statistic: str = "median",
	alternative: str = "two-sided", random_state: int = None
) -> tuple[float, float]:
    rng = numpy.random.default_rng(random_state)
    x, y = numpy.asarray(x), numpy.asarray(y)

    combined = numpy.concatenate([x, y])

    # Compute observed difference
    stat_func = numpy.mean if statistic == "mean" else numpy.median
    observed_diff = stat_func(x) - stat_func(y)

    # Null distribution
    perm_diffs = numpy.empty(n_permutations)
    for i in tqdm.tqdm(range(n_permutations)):
        perm = rng.permutation(combined)
        perm_x = perm[:len(x)]
        perm_y = perm[len(x):]
        perm_diffs[i] = stat_func(perm_x) - stat_func(perm_y)

    # Two-sided or one-sided p-value
    if alternative == "two-sided":
        p_value = numpy.mean(numpy.abs(perm_diffs) >= abs(observed_diff))
    elif alternative == "greater":
        p_value = numpy.mean(perm_diffs >= observed_diff)
    elif alternative == "less":
        p_value = numpy.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(
        	"alternative must be 'two-sided', 'greater', or 'less'")

    return p_value, observed_diff


def _plot_imput_impact_violin_sinle(axes: matplotlib.axes.Axes,
	values: numpy.ndarray | list, *,
	position: float, side: str, edgecolor: str, facecolor: str | None = None,
) -> dict:
	if facecolor is None:
		facecolor = edgecolor

	violin = axes.violinplot([values], positions=[position], widths=0.7,
		orientation="vertical", side=side, showextrema=False)
	for a in violin["bodies"]:
		a.set(edgecolor=edgecolor, facecolor=edgecolor + "40", alpha=None,
			linewidth=0.7, zorder=5)

	# add measn manually
	if side == "low":
		x = [position - 0.1, position - 0.02]
	elif side == "high":
		x = [position + 0.02, position + 0.1]
	else:
		x = [position - 0.1, position + 0.1]
	axes.plot(x, [numpy.median(values)] * 2, linewidth=2.0, color=edgecolor,
		zorder=6)

	return violin


def _imput_impact_sig_test(base_vals: numpy.ndarray,
	imput_data: dict[str, numpy.ndarray], *,
	conf_level: float = 0.05,
) -> dict[str, float]:
	keys = sorted(imput_data.keys())
	n_data = len(imput_data)
	pvalues = numpy.empty(n_data, dtype=float)
	for i, k in enumerate(keys):
		pvalues[i] = _permutation_test(base_vals, imput_data[k],
			alternative="two-sided")[0]

	# benjamini-hochberg procedure
	sort_idx = numpy.argsort(pvalues)
	bh_sig = numpy.zeros(n_data, dtype=bool)
	adj_conf = conf_level * (numpy.arange(n_data) + 1) / n_data
	bh_sig[sort_idx] = pvalues[sort_idx] <= adj_conf

	# summary
	ret = dict()
	for k, p, bh in zip(keys, pvalues, bh_sig):
		ret[k] = p if bh else numpy.nan
	return ret


def _get_sig_mark(adj_pvalue: float) -> tuple[str, str]:
	# return s (str) and color
	if adj_pvalue < 1e-3:
		s = "***"
		color = "#ff0000"
	elif adj_pvalue < 1e-2:
		s = "**"
		color = "#ff0000"
	elif adj_pvalue < 5e-2:
		s = "*"
		color = "#ff0000"
	else:
		s = "NS"
		color = "#000000"
	return s, color


def _expand_ylims(axes: matplotlib.axes.Axes, expand_frac: float) -> None:
	low, high = axes.get_ylim()
	expand = (high - low) * expand_frac
	axes.set_ylim(low - expand, high + expand)
	return


def _plot_ill_cpg_imput_impact_single(axes: matplotlib.axes.Axes, *,
	full_stat: dict,
	full_sig_res: dict,
	dataset: str,
	depth: int,
	imput_cfgs: list[ImputCfg],
	imput_impact: ImputImpactCfg,
	dataset_cfgs: pylib.DatasetCfgLib,
	show_xlabel: bool = False,
):
	# extract data
	_extracted = extract_imput_age_predict_stat(full_stat, depth)
	data = {k: v[imput_impact.key] for k, v in _extracted[dataset].items()}
	plain_vals = data["plain"]

	dataset_color = dataset_cfgs[dataset].color
	plain_color = "#9d999a"

	for i, v in enumerate(imput_cfgs):
		violin = _plot_imput_impact_violin_sinle(axes, plain_vals,
			position=i + 0.48, side="low", edgecolor=plain_color,
		)
		violin = _plot_imput_impact_violin_sinle(axes, data[v.key],
			position=i + 0.52, side="high", edgecolor=dataset_color,
		)

	if imput_impact.add_legend:
		handels = list()
		p = matplotlib.patches.Rectangle((0, 0), 0, 0, edgecolor=plain_color,
			facecolor=plain_color + "40", label="No imputation")
		handels.append(p)
		p = matplotlib.patches.Rectangle((0, 0), 0, 0, edgecolor=dataset_color,
			facecolor=dataset_color + "40", label="Imputed")
		handels.append(p)
		legend = axes.legend(handles=handels, loc=2, bbox_to_anchor=[1.02, 0.98],
			frameon=False, handlelength=0.75, fontsize=8)

	# misc
	axes.tick_params(
		left=True, labelleft=True,
		right=False, labelright=False,
		bottom=show_xlabel, labelbottom=show_xlabel,
		top=False, labeltop=False,
	)
	axes.set_xlim(0, len(imput_cfgs))
	# ylim upper and lower limits + 20%
	_expand_ylims(axes, 0.2)
	# fit dataset name in the upper extended spaces
	axes.text(0.50, 0.95, dataset_cfgs[dataset].display_name,
		transform=axes.transAxes,
		horizontalalignment="center", verticalalignment="top"
	)

	# significance test and add marker in the lower extended spaces
	sig_res = _imput_impact_sig_test(plain_vals, data, conf_level=0.05)
	ylim_low, ylim_high = axes.get_ylim()
	mark_y = ylim_low + (ylim_high - ylim_low) * 0.05
	for i, v in enumerate(imput_cfgs):
		s, color = _get_sig_mark(sig_res[v.key])
		fontweight = "bold" if "*" in s else None
		axes.text(i + 0.5, mark_y, s,
			color=color, fontsize=10, zorder=10, fontweight=fontweight,
			path_effects=[
				matplotlib.patheffects.Stroke(linewidth=3, foreground="#ffffffc0"),
			],
			horizontalalignment="center", verticalalignment="bottom",
		)
		axes.text(i + 0.5, mark_y, s,
			color=color, fontsize=10, zorder=11, fontweight=fontweight,
			horizontalalignment="center", verticalalignment="bottom",
		)
	axes.set_xticks(numpy.arange(len(imput_cfgs)) + 0.5)
	if show_xlabel:
		axes.set_xticklabels([v.display_name for v in imput_cfgs])
		axes.set_xlabel("Imputation methods")

	return


def _add_shared_ylabel(figure: matplotlib.figure.Figure, *,
	axes_list=list[matplotlib.axes.Axes],
	ylabel: str, **kw,
):
	bl_xys = [axes.transAxes.transform((-0.15, 0)) for axes in axes_list]
	tl_xys = [axes.transAxes.transform((-0.15, 1)) for axes in axes_list]
	px_x = min(v[0] for v in tl_xys)
	px_y = (min(v[1] for v in bl_xys) + max(v[1] for v in tl_xys)) / 2
	figure.text(*figure.transFigure.inverted().transform((px_x, px_y)), ylabel,
		rotation=90,
		horizontalalignment="right", verticalalignment="center"
	)
	return


def _plot_ill_cpg_imput_vs_depth(axes: matplotlib.axes.Axes, *,
	full_stat: dict[str, dict],
	datasets: list[str],
	dataset_cfgs: pylib.DatasetCfgLib,
	imput_cfg=ImputCfg,
	imput_impact=ImputImpactCfg,
	show_xlabel: bool = False,
	add_legend: bool = False,
):
	handles = list()
	for ds in datasets:
		# extract data
		plain_vals = full_stat[ds]["plain"][imput_impact.key]
		plain_mean = plain_vals.mean()

		# imput vals
		_vals = dict()
		for (imput, depth), v in full_stat[ds]["imput"].items():
			if imput == imput_cfg.key:
				_vals[depth] = v[imput_impact.key]

		# rearrange data
		x = [k for k in sorted(_vals.keys()) if k != 0]
		y = [_vals[i].mean() for i in x]
		x = numpy.asarray(x, dtype=int)
		y = numpy.asarray(y, dtype=float)

		# plot
		p = axes.scatter(x, y - plain_mean, zorder=10,
			marker="o", s=40, linewidths=0.5,
			edgecolors="#000000", facecolors=dataset_cfgs[ds].color + "80",
			label=dataset_cfgs[ds].display_name,
		)
		handles.append(p)

	# legend
	if add_legend:
		axes.legend(handles=handles, loc=9, bbox_to_anchor=[0.5, -0.45],
			handlelength=0.75, ncol=2,
		)

	# misc
	axes.axhline(0, color="#a0a0a0", linewidth=2)
	axes.grid(color="#d0d0d0")
	axes.tick_params(bottom=show_xlabel, labelbottom=show_xlabel)
	if show_xlabel:
		axes.set_xlabel(f"Max. {imput_cfg.display_name} imputation depth")
	axes.set_ylabel(
		f"Imput. $\\Delta${imput_impact.display_name}\n({imput_impact.unit})")
	axes.yaxis.set_label_coords(-0.18, 0.5)

	return


def _plot_ill_cpg_imput_impact(layout: dict, datasets: list[str], *,
	imput_impact_cfgs: list[dict[str, str]], imput_cfgs: list[ImputCfg],
	clocks: list[str], dataset_cfgs: pylib.DatasetCfgLib,
	depth: int = 0,
):
	# prepare data
	full_stat = prep_full_imput_age_predict_stat(datasets, clocks)
	# full-scale significance test with b-h procesdure
	full_sig_res = _full_sig_test_bh(full_stat)

	# imputation impact vs depth
	layout["imput_depth_mae"].sharex(layout["imput_depth_reprod"])

	_plot_ill_cpg_imput_vs_depth(layout["imput_depth_mae"],
		full_stat=full_stat,
		datasets=datasets,
		dataset_cfgs=dataset_cfgs,
		imput_cfg=imput_cfgs[2],  # knn
		imput_impact=imput_impact_cfgs[0],  # repl_std/repl_mad
		show_xlabel=False,
		add_legend=False,
	)

	_plot_ill_cpg_imput_vs_depth(layout["imput_depth_reprod"],
		full_stat=full_stat,
		datasets=datasets,
		dataset_cfgs=dataset_cfgs,
		imput_cfg=imput_cfgs[2],  # knn
		imput_impact=imput_impact_cfgs[1],  # repl_std/repl_mad
		show_xlabel=True,
		add_legend=True,
	)

	for ds, imput in itertools.product(datasets, imput_impact_cfgs):
		_plot_ill_cpg_imput_impact_single(
			layout[f"imput_impact_{ds}_{imput.key}"],
			full_stat=full_stat,
			full_sig_res=full_sig_res,
			dataset=ds,
			depth=depth,
			imput_impact=imput,
			imput_cfgs=imput_cfgs,
			dataset_cfgs=dataset_cfgs,
			show_xlabel=(ds is datasets[-1]),
		)

	for i, imput in enumerate(imput_impact_cfgs):
		# add y label
		_add_shared_ylabel(layout["figure"],
			axes_list=[layout[f"imput_impact_{ds}_{imput.key}"] for ds in datasets],
			ylabel=imput.ylabel or imput.display_name,
		)
		# add subplot label
		axes = layout[f"imput_impact_{datasets[0]}_{imput.key}"]

	return


def _main():
	# configs
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	ill_surv_datasets = datasets
	imput_impact_cfgs = [
		ImputImpactCfg(key="mae", display_name="MAE", unit="years"),
		ImputImpactCfg(key="repl_mad", display_name="RD", unit="years",
			add_legend=True),
	]
	imput_cfgs = [
		ImputCfg(key="mean", display_name="Mean"),
		ImputCfg(key="median", display_name="Median"),
		ImputCfg(key="knn", display_name="KNN"),
		ImputCfg(key="constant", display_name="0-filling"),
	]

	# load data
	dataset_cfgs = pylib.DatasetCfgLib.from_json()
	clocks = pylib.clock.load_clocks(exclude={"altumage"})

	# plot
	layout = setup_layout(
		ill_surv_datasets=ill_surv_datasets,
		imput_impact_datasets=datasets,
		imput_impact_cfgs=imput_impact_cfgs,
	)
	figure = layout["figure"]

	_plot_ill_cpg_survey(layout, ill_surv_datasets,
		clocks=clocks,
		dataset_cfgs=dataset_cfgs
	)

	_plot_ill_cpg_imput_impact(layout, datasets=datasets,
		imput_impact_cfgs=imput_impact_cfgs,
		imput_cfgs=imput_cfgs,
		clocks=clocks,
		dataset_cfgs=dataset_cfgs,
		depth=0,
	)

	figure.savefig("fig_4.svg", dpi=600)
	matplotlib.pyplot.close(figure)

	return


if __name__ == "__main__":
	with threadpoolctl.threadpool_limits(limits=24, user_api="blas"):
		_main()
