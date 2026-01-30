#!/usr/bin/env python3

import dataclasses
import pdb

import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.legend
import matplotlib.legend_handler
import matplotlib.patheffects
import matplotlib.pyplot
import matplotlib.text
import mpllayout

import pylib


def _load_and_prep_data(datasets: list[str], clocks: list[str],
) -> dict[str, dict[str]]:
	ret = dict()
	for ds in datasets:
		repl_group = pylib.dataset.ReplGroup.load_dataset(ds)

		ds_res: dict[str] = {
			"repl_group": repl_group,
		}
		for prep in ["full", "filtered"]:
			age_pred = pylib.age_pred_res.AgePredRes.from_txt(
				f"runs/{ds}/final.{prep}.age_pred.tsv"
			)
			ds_res[prep] = age_pred.calc_stat(clocks, repl_group=repl_group)

		ret[ds] = ds_res

	return ret


@dataclasses.dataclass
class SeriesConfig(object):
	key: str
	display_name: str
	marker: str = None
	primary: bool = False


def setup_layout(n_datasets: int, ncols: int = 2) -> dict:
	lc = mpllayout.LayoutCreator(
		left_margin=1.0,
		right_margin=2.0,
		bottom_margin=0.7,
		top_margin=0.5,
	)

	nrows = (n_datasets + ncols - 1) // ncols

	axes_s = 5.0
	axes_gap_w = 0.8
	axes_gap_h = 1.0

	for i in range(n_datasets):
		ri = nrows - i // ncols - 1
		ci = i % ncols

		axes = lc.add_frame(f"axes_{i}")
		axes.set_anchor("bottomleft", offsets=(
			(axes_s + axes_gap_w) * ci,
			(axes_s + axes_gap_h) * ri
		))
		axes.set_size(axes_s, axes_s)

	layout = lc.create_figure_layout()
	layout["nrows"] = nrows
	layout["ncols"] = ncols
	layout["legend_axes_id"] = ncols - 1

	return layout


class ClockMarkerHandler(matplotlib.legend_handler.HandlerPathCollection):
	def __init__(self, *ka, clock_cfgs: pylib.ClockCfgLib, **kw):
		self.clock_cfgs = clock_cfgs
		return super().__init__(*ka, **kw)

	def create_artists(self, legend, orig_handle: matplotlib.collections.PathCollection,
		xdescent: float, ydescent: float, width: float, height: float,
		fontsize: float, transform
	) -> list:
		artists = super().create_artists(legend, orig_handle,
			xdescent, ydescent, width, height, fontsize, transform)
		# overlay text
		s = self.clock_cfgs[orig_handle.get_label()].marker_char
		fontsize = 5 + 2 / len(s)
		t = matplotlib.text.Text(width / 2, height / 3, s,
			fontsize=fontsize, color="#000000",
			transform=transform, zorder=artists[0].zorder + 1,
			path_effects=[
				matplotlib.patheffects.Stroke(linewidth=1.5, foreground="#ffffff"),
			],
			horizontalalignment="center", verticalalignment="center",
		)
		artists.append(t)
		t = matplotlib.text.Text(width / 2, height / 3, s,
			fontsize=fontsize, color="#000000",
			transform=transform, zorder=artists[0].zorder + 1,
			horizontalalignment="center", verticalalignment="center",
		)
		artists.append(t)
		return artists


def plot_mae_err_scatter_dataset(axes: matplotlib.axes.Axes, *,
	age_pred_stat: dict[str], clocks: list[str], clock_cfgs: pylib.ClockCfgLib,
	series_cfgs: list[SeriesConfig], dataset_cfg: pylib.DatasetCfg,
	add_legend: bool = False,
):
	# extract data
	plot_data = dict()
	for s in series_cfgs:
		plot_data[s.key] = {
			"x": age_pred_stat[s.key]["mae"].values,
			"y": age_pred_stat[s.key]["repl_mad"].values,
		}
	clock_handles = list()
	for s in series_cfgs:
		for c, x, y in zip(clocks, plot_data[s.key]["x"], plot_data[s.key]["y"]):
			clock_cfg = clock_cfgs[c]
			p = axes.scatter([x], [y], marker=s.marker, s=100, linewidths=0.5,
				edgecolors="#000000", facecolors=clock_cfg.color,
				label=clock_cfg.key,
				zorder=(6 if s.key == "full" else 5),
			)
			if s.primary:
				clock_handles.append(p)
				# add clock text only for primary marker (legend)
				mch = clock_cfgs[c].marker_char
				fontsize = 5 + 2 / len(mch)
				axes.text(x, y, mch, fontsize=fontsize, zorder=10,
					path_effects=[
						matplotlib.patheffects.Stroke(linewidth=1.5,
						foreground="#ffffff"),
					],
					horizontalalignment="center", verticalalignment="center",
				)
				axes.text(x, y, mch, fontsize=fontsize, zorder=11,
					horizontalalignment="center", verticalalignment="center",
				)

	# add annotation
	for i, c in enumerate(clocks):
		axes.annotate("",
			xy=(plot_data[series_cfgs[1].key]["x"][i],
				plot_data[series_cfgs[1].key]["y"][i],
			),
			xytext=(
				plot_data[series_cfgs[0].key]["x"][i],
				plot_data[series_cfgs[0].key]["y"][i],
			),
			xycoords=axes.transData, textcoords=axes.transData,
			arrowprops=dict(arrowstyle="->", color=clock_cfgs[c].color,
				shrinkA=6, shrinkB=6, linewidth=0.5),
		)

	if add_legend:
		# imput legend
		imput_handles = list()
		for s in series_cfgs:
			p = axes.scatter([], [], marker=s.marker, s=100, linewidths=0.5,
				edgecolors="#000000", facecolors="#FFFFFF",
				label=s.display_name,
			)
			imput_handles.append(p)
		legend = axes.legend(handles=imput_handles,
			loc=2, bbox_to_anchor=[1.02, 1.02], frameon=False,
			handlelength=0.75, title="Imputation", title_fontsize=12,
		)
		axes.add_artist(legend)

		# clock legend
		handler_map = matplotlib.legend.Legend.get_default_handler_map().copy()
		handler_map[matplotlib.collections.PathCollection] = ClockMarkerHandler(
			clock_cfgs=clock_cfgs
		)
		legend = axes.legend(handles=clock_handles, handler_map=handler_map,
			loc=2, bbox_to_anchor=[1.02, 0.82], frameon=False,
			handlelength=0.75, title="Clocks", title_fontsize=12,
		)

	# misc
	axes.set_xlabel("MAE (years)")
	axes.set_ylabel("RD (years)")
	axes.set_title(dataset_cfg.display_name)

	return


def _main():
	# below is order-related
	datasets = ["RB_GDNA_GALAXY", "RB_GDNA_TWIST", "RB_GALAXY", "RB_TWIST"]
	n_datasets = len(datasets)

	# load data
	clocks = pylib.clock.load_clocks()
	clocks = [c for c in clocks if c not in {"altumage"}]
	clock_cfgs = pylib.ClockCfgLib.from_json()
	data = _load_and_prep_data(datasets, clocks)
	dataset_cfgs = pylib.DatasetCfgLib.from_json()

	# plot config
	series_cfgs = [
		SeriesConfig(key="full", display_name="Not imputed", marker="o",
			primary=True),
		SeriesConfig(key="filtered", display_name="Imputed (KNN)", marker="^"),
	]

	layout = setup_layout(n_datasets)
	figure = layout["figure"]

	for i, ds in enumerate(datasets):
		plot_mae_err_scatter_dataset(layout[f"axes_{i}"],
			age_pred_stat=data[ds],
			clocks=clocks,
			clock_cfgs=clock_cfgs,
			series_cfgs=series_cfgs,
			dataset_cfg=dataset_cfgs[ds],
			add_legend=(i == layout["legend_axes_id"])
		)

	figure.savefig("plot_fig_s15.svg", dpi=600)
	figure.savefig("plot_fig_s15.png", dpi=600)
	figure.savefig("plot_fig_s15.pdf", dpi=600)
	matplotlib.pyplot.close(figure)
	return


if __name__ == "__main__":
	_main()
