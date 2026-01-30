#!/usr/bin/env python3

import dataclasses
from typing import Sequence

import matplotlib
import matplotlib.pyplot
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import mpllayout
import numpy
import pandas
import tqdm

import pylib

matplotlib.rcParams["font.size"] = 8


@dataclasses.dataclass
class CpGSeriesCfg(object):
    key: str
    display_name: str
    select: Sequence[str] | None = None
    facecolor: str = "#ffffff"
    method: str = "scatter"


def setup_layout(datasets: Sequence[str], series_cfgs: Sequence[CpGSeriesCfg],
                 ) -> dict[str]:
    lc = mpllayout.LayoutCreator(
        left_margin=1.0,
        right_margin=0.2,
        bottom_margin=0.5,
        top_margin=0.3,
    )

    n_blocks = len(series_cfgs)
    n_datasets = len(datasets)

    n_row = int(numpy.sqrt(n_datasets))
    n_col = (n_datasets + n_row - 1) // n_row
    block_gap_h = 0.7

    cell_w = 1.5
    cell_h = cell_w
    cell_gap_w = 0.3
    cell_gap_h = 0.3

    axes_pos: dict[str, tuple[int, int]] = dict()
    for _ib, series_cfg in enumerate(series_cfgs):
        ib = n_blocks - 1 - _ib

        block_offset_h = cell_h * n_row + \
            cell_gap_h * (n_row - 1) + block_gap_h

        for _i, dataset in enumerate(datasets):
            ir = n_row - _i // n_col - 1
            ic = _i % n_col

            # add axes
            name = f"cell_{series_cfg.key}_{dataset}"
            axes = lc.add_frame(name)
            axes.set_anchor("bottomleft", offsets=(
                (cell_w + cell_gap_w) * ic,
                (cell_h + cell_gap_h) * ir + block_offset_h * ib,
            ))
            axes.set_size(cell_w, cell_h)
            axes_pos[name] = (ir, ic)

    layout = lc.create_figure_layout()
    layout["n_row"] = n_row
    layout["n_col"] = n_col
    layout["axes_pos"] = axes_pos

    return layout


def _plot_dataset_single(axes: matplotlib.axes.Axes, *,
                         data: pandas.DataFrame, dataset: str, dataset_cfg: pylib.DatasetCfg,
                         method: str = "scatter", series_cfg: CpGSeriesCfg,
                         show_xlabel: bool = False, show_ylabel: bool = False,
                         ) -> None:

    dataset_mask = data["Dataset"] == dataset
    x = data.loc[dataset_mask, "Mean"]
    y = data.loc[dataset_mask, "ICC"]

    if method == "scatter":
        axes.scatter(x, y, marker="o", s=6, edgecolors="#000000",
                     facecolors="#000000", zorder=10,
                     )
        axes.scatter(x, y, marker="o", s=5, edgecolors="none",
                     facecolors="#ffffff", zorder=15,
                     )
        axes.scatter(x, y, marker="o", s=5, edgecolors="none",
                     facecolors=series_cfg.facecolor, zorder=20,
                     )
    elif method == "heatmap":
        xbins = numpy.linspace(-1, 1, 200)
        ybins = numpy.linspace(-1, 1, 200)
        hist, xedges, yedges = numpy.histogram2d(x, y, bins=[xbins, ybins],
                                                 density=True)

        axes.pcolor(xedges, yedges, hist.T, cmap="viridis", rasterized=True,
                    vmin=0, vmax=numpy.percentile(hist, 99.7), zorder=10,
                    )
    else:
        raise ValueError(f"unknown plot method: {method}")

    # misc
    axes.tick_params(
        left=True, labelleft=show_ylabel,
        right=False, labelright=False,
        bottom=True, labelbottom=show_xlabel,
        top=False, labeltop=False,
    )
    axes.grid(linestyle="-", color="#e0e0e0", zorder=0)
    axes.set(xlim=(0, 1), ylim=(-1, 1))
    if show_xlabel:
        axes.set_xlabel("Beta-value")
    if show_ylabel:
        axes.set_ylabel(f"ICC")
    axes.set_title(dataset_cfg.display_name)

    return


def _plot_series(layout: dict, data: pandas.DataFrame, series_cfg: CpGSeriesCfg,
                 *, datasets: Sequence[str], dataset_cfgs: pylib.DatasetCfgLib,
                 ) -> None:
    assert len(datasets) > 0

    # series data preprocessing
    if series_cfg.select is not None:
        series_data = data[data["CpG"].isin(series_cfg.select)].copy()
    else:
        series_data = data.copy()

    figure: matplotlib.figure.Figure = layout["figure"]

    # plot by dataset
    ref_axes = None
    topleft_axes = None
    bottomleft_axes = None

    for dataset in tqdm.tqdm(datasets):
        dataset_cfg = dataset_cfgs[dataset]

        axes_name = f"cell_{series_cfg.key}_{dataset}"
        axes: matplotlib.axes.Axes = layout[axes_name]
        ir, ic = layout["axes_pos"][axes_name]

        if ref_axes is None:
            ref_axes = axes
        else:
            axes.sharex(ref_axes)
            axes.sharey(ref_axes)

        if (ic == 0) and (ir == 0):
            bottomleft_axes = axes
        if (ic == 0) and (ir == layout["n_row"] - 1):
            topleft_axes = axes

        _plot_dataset_single(axes,
                             data=series_data,
                             dataset=dataset,
                             dataset_cfg=dataset_cfg,
                             method=series_cfg.method,
                             series_cfg=series_cfg,
                             show_xlabel=(ir == 0),
                             show_ylabel=(ic == 0),
                             )

    # add series title line
    xy_bottom = figure.transFigure.inverted().transform(
        bottomleft_axes.transAxes.transform((-0.4, 0.0))
    )
    xy_top = figure.transFigure.inverted().transform(
        topleft_axes.transAxes.transform((-0.4, 1.0))
    )
    line = matplotlib.lines.Line2D(
        [xy_bottom[0], xy_top[0]],
        [xy_bottom[1], xy_top[1]],
        transform=figure.transFigure,
        color="#000000",
        linewidth=1.5,
        zorder=20,
    )
    figure.add_artist(line)
    xy_text = ((xy_bottom[0] + xy_top[0]) / 2, (xy_bottom[1] + xy_top[1]) / 2)
    figure.text(*xy_text, series_cfg.display_name, rotation=90, fontsize=10,
                horizontalalignment="right", verticalalignment="center", zorder=25,
                )

    return


def _main():
    # configs
    datasets = ["MSA", "EPICv2", "cfDNAGalaxy", "cfDNATwist", "gDNAGalaxy",
                "gDNATwist"]

    series_cfgs = [
        CpGSeriesCfg(
            key="method-shared",
            display_name="Method-shared CpGs",
            select=None,
            facecolor="#519ef601",
            method="heatmap",
        ),
        CpGSeriesCfg(
            key="clock-shared",
            display_name="Clock-shared CpGs",
            select=open("2727_clock_cpg.txt", "r").read().splitlines(),
            facecolor="#519ef660",
            method="scatter",
        ),
    ]

    # load data
    data = pandas.read_csv("data/probeICC.txt", sep="\t")
    dataset_cfgs = pylib.DatasetCfgLib.from_json()

    layout = setup_layout(datasets, series_cfgs)
    figure: matplotlib.figure.Figure = layout["figure"]

    # plot
    for series_cfg in series_cfgs:
        _plot_series(layout, data, series_cfg,
                     datasets=datasets,
                     dataset_cfgs=dataset_cfgs,
                     )

    # save
    figure.savefig("fig_s2.svg", dpi=600)
    figure.savefig("fig_s2.png", dpi=600)
    figure.savefig("fig_s2.pdf", dpi=600)
    matplotlib.pyplot.close(figure)
    return


if __name__ == "__main__":
    _main()
