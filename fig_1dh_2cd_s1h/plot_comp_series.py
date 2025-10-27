#!/usr/bin/env python3

import dataclasses
from typing import Callable, Sequence

import matplotlib
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot
import mpllayout
import numpy
import pandas
import scipy.stats
import statsmodels
import statsmodels.stats.multitest
import tqdm

import pylib


@dataclasses.dataclass
class SigCfg(object):
	mark: str
	patch_color: str = "#a0a0a0"
	font_color: str = "#000000"
	font_size: float = None
	p_min: float = -float("inf")
	p_max: float = float("inf")

	@property
	def label(self) -> str:
		if (self.p_min == -float("inf")) and (self.p_max == float("inf")):
			ret = "any"
		elif self.p_min == -float("inf"):
			ret = f"p ≤ {str(self.p_max)}"
		elif self.p_max == float("inf"):
			ret = f"p > {str(self.p_min)}"
		else:
			ret = f"{str(self.p_min)} < p ≤ {str(self.p_max)}"
		return ret

	def bh_mask(self, p_values: numpy.ndarray[float]) -> numpy.ndarray[bool]:
		p_min_reject, *_ = statsmodels.stats.multitest.multipletests(
			p_values, alpha=self.p_min, method="fdr_bh")
		p_max_reject, *_ = statsmodels.stats.multitest.multipletests(
			p_values, alpha=self.p_max, method="fdr_bh")
		return p_max_reject & (~p_min_reject)


def bh_procedure(p_values: numpy.ndarray[float], sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	n = len(p_values)
	# b-h procedure
	sig_vec = numpy.empty(n, dtype=object)
	for cfg in sig_cfgs:
		mask = cfg.bh_mask(p_values)
		sig_vec[mask] = cfg
	return sig_vec


def load_icc_data_series(ifname: str, *,
	row_field: str, col_field: str, value_field: str,
	select_file: str = None,
) -> pandas.DataFrame:
	data = pandas.read_csv(ifname, sep="\t")
	pivot_table = data.pivot(index=row_field, columns=col_field,
		values=value_field)
	if select_file is not None:
		with open(select_file) as fp:
			select = fp.read().splitlines()
		pivot_table = pivot_table.reindex(columns=select)
	return pivot_table


def setup_layout(n_datasets: int) -> dict[str]:
	lc = mpllayout.LayoutCreator(
		left_margin=0.9,
		right_margin=0.2,
		bottom_margin=1.0,
		top_margin=0.2,
	)

	dataset_sp = 0.32

	box_w = dataset_sp * n_datasets
	box_h = 1.8
	box_gap_h = 0.1

	grid_w = box_w
	grid_h = dataset_sp * (n_datasets / 2 + 0.25)

	# add frames
	box = lc.add_frame("box")
	box.set_anchor("bottomleft")
	box.set_size(box_w, box_h)

	grid = lc.add_frame("grid")
	grid.set_anchor("bottomleft", box, "topleft", offsets=(0, box_gap_h))
	grid.set_size(grid_w, grid_h)

	layout = lc.create_figure_layout()

	return layout


def _add_boxplot(axes: matplotlib.axes.Axes, data: pandas.DataFrame, *,
	datasets: list[str], dataset_cfgs: pylib.DatasetCfgLib,
	with_scatter: bool = False, ylabel: str = None,
):
	n_datasets = len(datasets)

	for i, ds in enumerate(datasets):
		vals = data.loc[ds].copy().dropna().values
		cfg = dataset_cfgs[ds]

		box = axes.boxplot(vals, positions=[i + 0.5], widths=0.9,
			patch_artist=True, showfliers=False,
			boxprops=dict(linewidth=0.7, edgecolor=cfg.color,
				facecolor="#ffffff" if with_scatter else cfg.color + "20",),
			medianprops=dict(linewidth=1.2, color=cfg.color),
			whiskerprops=dict(linewidth=0.7, color=cfg.color),
			capprops=dict(linewidth=0.7, color=cfg.color),
			capwidths=0.75,
			zorder=5,
		)

		# add scatter
		if not with_scatter:
			continue

		# randomize x scaled by local density
		# estimat density using kde
		kde = scipy.stats.gaussian_kde(vals)
		scale = kde(vals)
		if (s_max := scale.max()) > 0:
			scale /= s_max

		x = numpy.random.normal(i + 0.5, scale / 10, size=len(vals))
		axes.scatter(x, vals, s=30, linewidths=0.25, edgecolors="#000000",
			facecolors=cfg.color + "40", zorder=10)

	# misc
	for sp in axes.spines.values():
		sp.set_linewidth(0.5)
	axes.set_xlim(0, n_datasets)
	axes.set_ylabel(ylabel, fontsize=10)
	axes.set_xticks(numpy.arange(n_datasets) + 0.5)
	axes.set_xticklabels([dataset_cfgs[ds].display_name for ds in datasets],
		rotation=40,
		horizontalalignment="right", verticalalignment="top")

	return


def _linear_idx_to_triag(idx: int, n: int) -> tuple[int, int]:
	if (idx > n * (n - 1) // 2) or (idx < 0):
		raise ValueError("index out of range")

	for r in range(n - 1):
		if idx < (n - r - 1):
			c = r + 1 + idx
			break
		else:
			idx -= (n - r - 1)
	return r, c


def block_permutation_test(x, y, *, n_permutations=10000, random_state=None):
    rng = numpy.random.default_rng(random_state)
    diffs = x - y
    observed_stat = numpy.mean(diffs)

    # Generate permutations: randomly flip sign of differences
    perm_stats = []
    for _ in tqdm.tqdm(range(n_permutations)):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_stats.append(numpy.mean(diffs * signs))
    perm_stats = numpy.array(perm_stats)

    # Compute p-value
    p_value = numpy.mean(numpy.abs(perm_stats) >= abs(observed_stat))

    return observed_stat, p_value


def _sig_test_from_pairwise(data: pandas.DataFrame, datasets: Sequence[str],
	meth_func: Callable[[pandas.Series, pandas.Series], float],
) -> numpy.ndarray[float]:
	n_datasets = len(datasets)
	n_pairs = n_datasets * (n_datasets - 1) // 2
	# calculate p values
	p_values = numpy.empty(n_pairs, dtype=float)
	for i in range(n_pairs):
		r, c = _linear_idx_to_triag(i, n_datasets)
		s1 = data.loc[datasets[r]]
		s2 = data.loc[datasets[c]]
		mask = s1.notna() & s2.notna()  # drop na
		s1 = s1[mask]
		s2 = s2[mask]
		p_values[i] = meth_func(s1, s2)
	return p_values


def _sig_test_ttest_ind(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	p_values = _sig_test_from_pairwise(data, datasets,
		lambda x, y: scipy.stats.ttest_ind(x, y, alternative="two-sided",
			equal_var=False)[1]
	)
	return bh_procedure(p_values, sig_cfgs)


def _sig_test_ttest_rel(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	p_values = _sig_test_from_pairwise(data, datasets,
		lambda x, y: scipy.stats.ttest_rel(x, y, alternative="two-sided")[1]
	)
	return bh_procedure(p_values, sig_cfgs)


def _sig_test_mannwhitneyu(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	p_values = _sig_test_from_pairwise(data, datasets,
		lambda x, y: scipy.stats.mannwhitneyu(x, y, alternative="two-sided")[1]
	)
	return bh_procedure(p_values, sig_cfgs)


def _sig_test_wilcoxon(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	p_values = _sig_test_from_pairwise(data, datasets,
		lambda x, y: scipy.stats.wilcoxon(x, y, alternative="two-sided")[1]
	)
	return bh_procedure(p_values, sig_cfgs)


def _sig_test_block_permutation(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg],
) -> numpy.ndarray[SigCfg]:
	p_values = _sig_test_from_pairwise(data, datasets,
		lambda x, y: block_permutation_test(x.values, y.values)[1]
	)
	return bh_procedure(p_values, sig_cfgs)


def _sig_test(data: pandas.DataFrame, datasets: Sequence[str],
	sig_cfgs: list[SigCfg], *, method: str = "ttest_ind",
) -> numpy.ndarray[SigCfg]:
	n_datasets = len(datasets)
	n_pairs = n_datasets * (n_datasets - 1) // 2

	# raw p value vector calculation dispatch
	if method == "ttest_ind":
		sig_vec = _sig_test_ttest_ind(data, datasets, sig_cfgs)
	elif method == "ttest_rel":
		sig_vec = _sig_test_ttest_rel(data, datasets, sig_cfgs)
	elif method == "mannwhitneyu":
		sig_vec = _sig_test_mannwhitneyu(data, datasets, sig_cfgs)
	elif method == "wilcoxon":
		sig_vec = _sig_test_wilcoxon(data, datasets, sig_cfgs)
	elif method == "block_permutation":
		sig_vec = _sig_test_block_permutation(data, datasets, sig_cfgs)
	# elif method == "manova_tukey":
	# sig_vec = _sig_test_manova_tukey(data, datasets, sig_cfgs)
	# elif method == "permanova":
	# sig_vec = _sig_test_permanova(data, datasets, sig_cfgs)
	else:
		raise ValueError(f"unsupported method: {method}")

	# transform back to matrix
	sig_matrix = numpy.empty((n_datasets, n_datasets), dtype=object)
	for i in range(n_pairs):
		r, c = _linear_idx_to_triag(i, n_datasets)
		sig_matrix[r, c] = sig_vec[i]
		sig_matrix[c, r] = sig_vec[i]

	return sig_matrix


def _add_sig_grid(axes: matplotlib.axes.Axes, data: pandas.DataFrame, *,
	datasets: list[str], dataset_cfgs: pylib.DatasetCfgLib,
	test_method: str,
	sig_cfgs: list[SigCfg],
	connection_axes=matplotlib.axes.Axes,
):
	n_datasets = len(datasets)

	# run correlation
	sig_matrix = _sig_test(data, datasets, sig_cfgs, method=test_method)
	pw = 0.48

	for r in range(0, n_datasets):
		for c in range(r + 1, n_datasets):
			sig = sig_matrix[r, c]
			x = (r + c) / 2 + 0.5
			y = (c - r) / 2
			path = [(x - pw, y), (x, y - pw), (x + pw, y), (x, y + pw)]
			# background mask
			p = matplotlib.patches.Polygon(path, closed=True, clip_on=False,
				linewidth=0.5, zorder=4,
				edgecolor="#ffffff", facecolor="#ffffff",
			)
			axes.add_patch(p)
			# foreground patch
			p = matplotlib.patches.Polygon(path, closed=True, clip_on=False,
				linewidth=0.5, zorder=5,
				edgecolor=sig.patch_color, facecolor=sig.patch_color + "80",
			)
			axes.add_patch(p)
			axes.text(x, y - 0.05, sig.mark, zorder=10,
			 	color=sig.font_color, fontsize=sig.font_size,
				horizontalalignment="center", verticalalignment="center",
			)

	# add connector and lines
	for i, ds in enumerate(datasets):
		cfg = dataset_cfgs[ds]
		if i > 0:
			# left circle
			p = matplotlib.patches.Circle((i + pw / 2, 0.5 - pw / 2), 0.1,
				linewidth=1.0, edgecolor=cfg.color, facecolor="#ffffff",
				zorder=3,
			)
			axes.add_patch(p)
			# left line
			line = matplotlib.lines.Line2D([i + 0.5, i], [0, 0.5],
				clip_on=False, linewidth=1.0, color=cfg.color, zorder=2
			)
			axes.add_artist(line)
		if i < (n_datasets - 1):
			# right circle
			p = matplotlib.patches.Circle((i + 0.5 + pw / 2, 0.5 - pw / 2), 0.1,
				linewidth=1.0, edgecolor=cfg.color, facecolor="#ffffff",
				zorder=3,
			)
			axes.add_patch(p)
			# right line
			line = matplotlib.lines.Line2D([i + 0.5, i + 1], [0, 0.5],
				clip_on=False, linewidth=1.0, color=cfg.color, zorder=2
			)
			axes.add_artist(line)

		# connection line
		p = matplotlib.patches.ConnectionPatch(
			xyA=(i + 0.5, 0), coordsA=axes.transData,
			xyB=((i + 0.5) / n_datasets, 1.01), coordsB=connection_axes.transAxes,
			linewidth=1.0, color=cfg.color, zorder=1,
		)
		axes.add_patch(p)

	# misc
	axes.tick_params(
		left=False, labelleft=False,
		right=False, labelright=False,
		bottom=False, labelbottom=False,
		top=False, labeltop=False,
	)
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_xlim(0, n_datasets)
	axes.set_ylim(0, n_datasets / 2)

	return


def _plot(data_file: str, plot_file: str, *,
	# data extraction
	row_field: str, col_field: str, value_field: str,
	select_file: str = None,
	# significance test
	test_method: str = "ttest_ind",
	# plot elements
	with_scatter: bool = False,
	ylabel: str = None,
	skip: bool = False,
) -> None:
	if skip:
		pylib.logger.info(f"skipping {plot_file}")
		return

	# config
	numpy.random.seed(42)
	dataset_order = {v: i for i, v in enumerate(["MSA", "EPICv2", "gDNAGalaxy",
		"gDNATwist", "cfDNAGalaxy", "cfDNATwist"])}
	sig_cfgs = [
		SigCfg("***", patch_color="#a64bd3", font_color="#000000",
			p_max=0.001),
		SigCfg("**", patch_color="#5751c9", font_color="#000000",
			p_min=0.001, p_max=0.01),
		SigCfg("*", patch_color="#4da6d3", font_color="#000000",
			p_min=0.01, p_max=0.05),
		SigCfg("N.S.", patch_color="#c0c0c0", font_color="#c0c0c0",
			font_size=8, p_min=0.05),
	]

	# load data
	data = load_icc_data_series(data_file,
		row_field=row_field, col_field=col_field, value_field=value_field,
		select_file=select_file,
	)
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# datasets rearrange
	datasets = sorted(data.index, key=lambda x: dataset_order[x])
	n_datasets = len(datasets)

	# plot
	layout = setup_layout(n_datasets)
	figure = layout["figure"]

	_add_boxplot(layout["box"], data, datasets=datasets,
		dataset_cfgs=dataset_cfgs, with_scatter=with_scatter, ylabel=ylabel,
	)

	_add_sig_grid(layout["grid"], data, datasets=datasets,
		dataset_cfgs=dataset_cfgs, test_method=test_method, sig_cfgs=sig_cfgs,
		connection_axes=layout["box"],
	)

	figure.savefig(plot_file, dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	# fig 1d
	_plot(
		"fig_1d_s1h.cpg_icc.txt",
		f"fig_1d.pdf",
		row_field="Dataset",
		col_field="CpG",
		value_field="ICC",
		test_method="block_permutation",
		with_scatter=False,
		ylabel="ICC\n(Method-shared CpGs)",
	)
	# fig 1h
	_plot(
		"fig_1h.age_icc.txt",
		f"fig_1h.pdf",
		row_field="Dataset",
		col_field="Model",
		value_field="ICC",
		test_method="block_permutation",
		with_scatter=True,
		ylabel="ICC\n(All 53 models)",
	)
	# fig s1h
	_plot(
		"fig_1d_s1h.cpg_icc.txt",
		f"fig_s1h.pdf",
		row_field="Dataset",
		col_field="CpG",
		value_field="ICC",
		test_method="block_permutation",
		select_file="clock_shared_cpgs.2727.txt",
		with_scatter=False,
		ylabel="ICC\n(Clock-shared CpGs)",
	)
	# fig 2c
	_plot(
		"fig_2cd.r_mae.txt",
		f"fig_2c.pdf",
		row_field="Dataset",
		col_field="Model",
		value_field="R",
		test_method="block_permutation",
		select_file="fig_2cd.clocks.txt",
		with_scatter=True,
		ylabel="R\n(Hi-Cov-All)",
	)
	# fig 2d
	_plot(
		"fig_2cd.r_mae.txt",
		f"fig_2d.pdf",
		row_field="Dataset",
		col_field="Model",
		value_field="MAE",
		test_method="block_permutation",
		select_file="fig_2cd.clocks.txt",
		with_scatter=True,
		ylabel="MAE\n(Hi-Cov-All)",
	)
