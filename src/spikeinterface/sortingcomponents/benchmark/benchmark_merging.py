from __future__ import annotations

from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import (
    plot_agreement_matrix,
)

import matplotlib.patches as mpatches

# from spikeinterface.postprocessing import get_template_extremum_channel
from spikeinterface.core import get_noise_levels

import pylab as plt
import numpy as np

from .benchmark_tools import BenchmarkStudy, Benchmark

class MergingBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, sorting, exhaustive_gt=True):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.params = params
        self.sorting = sorting
        self.exhaustive_gt = exhaustive_gt
        self.result = {}

    def run(self, **job_kwargs):
        from spikeinterface.sortingcomponents.clustering.clustering_tools import final_cleaning_circus
        final_sorting = final_cleaning_circus(self.recording, self.sorting, **self.params)
        self.result["final_sorting"] = final_sorting

    def compute_result(self, **result_params):
        
        # self.result["no_merging"] = GroundTruthComparison(
        #     self.gt_sorting, self.sorting, exhaustive_gt=self.exhaustive_gt
        # )

        self.result["merging"] = GroundTruthComparison(
            self.gt_sorting, self.result["final_sorting"], exhaustive_gt=self.exhaustive_gt
        )

    _run_key_saved = [("final_sorting", "sorting")]

    _result_key_saved = [
        ("merging", "pickle"),
    ]


class MergingStudy(BenchmarkStudy):

    benchmark_class = MergingBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = MergingBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def get_count_units(self, case_keys=None, well_detected_score=None, redundant_score=None, overmerged_score=None):
        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if isinstance(case_keys[0], str):
            index = pd.Index(case_keys, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(case_keys, names=self.levels)

        columns = ["num_gt", "num_sorter", "num_well_detected"]
        comp = self.get_result(case_keys[0])["merging"]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_redundant", "num_overmerged", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)

        for key in case_keys:
            comp = self.get_result(key)["merging"]
            assert comp is not None, "You need to do study.run_comparisons() first"

            gt_sorting = comp.sorting1
            sorting = comp.sorting2

            count_units.loc[key, "num_gt"] = len(gt_sorting.get_unit_ids())
            count_units.loc[key, "num_sorter"] = len(sorting.get_unit_ids())
            count_units.loc[key, "num_well_detected"] = comp.count_well_detected_units(well_detected_score)

            if comp.exhaustive_gt:
                count_units.loc[key, "num_redundant"] = comp.count_redundant_units(redundant_score)
                count_units.loc[key, "num_overmerged"] = comp.count_overmerged_units(overmerged_score)
                count_units.loc[key, "num_false_positive"] = comp.count_false_positive_units(redundant_score)
                count_units.loc[key, "num_bad"] = comp.count_bad_units()

        return count_units

    def plot_unit_counts(self, case_keys=None, figsize=None, **extra_kwargs):
        from spikeinterface.widgets.widget_list import plot_study_unit_counts

        plot_study_unit_counts(self, case_keys, figsize=figsize, **extra_kwargs)

    def plot_agreements(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["merging"], ax=ax)

    def plot_performances_vs_snr(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=figsize)

        for count, k in enumerate(("accuracy", "recall", "precision")):

            ax = axs[count]
            for key in case_keys:
                label = self.cases[key]["label"]

                analyzer = self.get_sorting_analyzer(key)
                metrics = analyzer.get_extension("quality_metrics").get_data()
                x = metrics["snr"].values
                y = self.get_result(key)["merging"].get_performance()[k].values
                ax.scatter(x, y, marker=".", label=label)
                ax.set_title(k)

            if count == 2:
                ax.legend()

    # def plot_error_metrics(self, metric="cosine", case_keys=None, figsize=(15, 5)):

    #     if case_keys is None:
    #         case_keys = list(self.cases.keys())

    #     fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

    #     for count, key in enumerate(case_keys):

    #         result = self.get_result(key)
    #         scores = result["merging"].get_ordered_agreement_scores()

    #         unit_ids1 = scores.index.values
    #         unit_ids2 = scores.columns.values
    #         inds_1 = result["merging"].sorting1.ids_to_indices(unit_ids1)
    #         inds_2 = result["merging"].sorting2.ids_to_indices(unit_ids2)
    #         t1 = result["sliced_gt_templates"].templates_array[:]
    #         t2 = result["clustering_templates"].templates_array[:]
    #         a = t1.reshape(len(t1), -1)[inds_1]
    #         b = t2.reshape(len(t2), -1)[inds_2]

    #         import sklearn

    #         if metric == "cosine":
    #             distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
    #         else:
    #             distances = sklearn.metrics.pairwise_distances(a, b, metric)

    #         im = axs[0, count].imshow(distances, aspect="auto")
    #         axs[0, count].set_title(metric)
    #         fig.colorbar(im, ax=axs[0, count])
    #         label = self.cases[key]["label"]
    #         axs[0, count].set_title(label)

    # def plot_metrics_vs_snr(self, metric="agreement", case_keys=None, figsize=(15, 5)):

    #     if case_keys is None:
    #         case_keys = list(self.cases.keys())

    #     fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

    #     for count, key in enumerate(case_keys):

    #         result = self.get_result(key)
    #         scores = result["gt_comparison"].agreement_scores

    #         analyzer = self.get_sorting_analyzer(key)
    #         metrics = analyzer.get_extension("quality_metrics").get_data()

    #         unit_ids1 = result["gt_comparison"].unit1_ids
    #         matched_ids2 = result["gt_comparison"].hungarian_match_12.values
    #         mask = matched_ids2 > -1

    #         inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1[mask])
    #         inds_2 = result["gt_comparison"].sorting2.ids_to_indices(matched_ids2[mask])

    #         t1 = result["sliced_gt_templates"].templates_array[:]
    #         t2 = result["clustering_templates"].templates_array[:]
    #         a = t1.reshape(len(t1), -1)
    #         b = t2.reshape(len(t2), -1)

    #         import sklearn

    #         if metric == "cosine":
    #             distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
    #         elif metric == "l2":
    #             distances = sklearn.metrics.pairwise_distances(a, b, metric)

    #         snr_matched = metrics["snr"][unit_ids1[mask]]
    #         snr_missed = metrics["snr"][unit_ids1[~mask]]

    #         to_plot = []
    #         if metric in ["cosine", "l2"]:
    #             for found, real in zip(inds_2, inds_1):
    #                 to_plot += [distances[real, found]]
    #         elif metric == "agreement":
    #             for found, real in zip(matched_ids2[mask], unit_ids1[mask]):
    #                 to_plot += [scores.at[real, found]]
    #         axs[0, count].plot(snr_matched, to_plot, ".", label="matched")
    #         axs[0, count].plot(snr_missed, np.zeros(len(snr_missed)), ".", c="r", label="missed")
    #         axs[0, count].set_xlabel("snr")
    #         axs[0, count].set_ylabel(metric)
    #         label = self.cases[key]["label"]
    #         axs[0, count].set_title(label)
    #         axs[0, count].legend()

    # def plot_metrics_vs_depth_and_snr(self, metric="agreement", case_keys=None, figsize=(15, 5)):

    #     if case_keys is None:
    #         case_keys = list(self.cases.keys())

    #     fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

    #     for count, key in enumerate(case_keys):

    #         result = self.get_result(key)
    #         scores = result["gt_comparison"].agreement_scores

    #         positions = result["sliced_gt_sorting"].get_property('gt_unit_locations')
    #         #positions = self.datasets[key[1]][1].get_property("gt_unit_locations")
    #         depth = positions[:, 1]

    #         analyzer = self.get_sorting_analyzer(key)
    #         metrics = analyzer.get_extension("quality_metrics").get_data()

    #         unit_ids1 = result["gt_comparison"].unit1_ids
    #         matched_ids2 = result["gt_comparison"].hungarian_match_12.values
    #         mask = matched_ids2 > -1

    #         inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1[mask])
    #         inds_2 = result["gt_comparison"].sorting2.ids_to_indices(matched_ids2[mask])

    #         t1 = result["sliced_gt_templates"].templates_array[:]
    #         t2 = result["clustering_templates"].templates_array[:]
    #         a = t1.reshape(len(t1), -1)
    #         b = t2.reshape(len(t2), -1)

    #         import sklearn

    #         if metric == "cosine":
    #             distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
    #         elif metric == "l2":
    #             distances = sklearn.metrics.pairwise_distances(a, b, metric)

    #         snr_matched = metrics["snr"][unit_ids1[mask]]
    #         snr_missed = metrics["snr"][unit_ids1[~mask]]
    #         depth_matched = depth[mask]
    #         depth_missed = depth[~mask]

    #         to_plot = []
    #         if metric in ["cosine", "l2"]:
    #             for found, real in zip(inds_2, inds_1):
    #                 to_plot += [distances[real, found]]
    #         elif metric == "agreement":
    #             for found, real in zip(matched_ids2[mask], unit_ids1[mask]):
    #                 to_plot += [scores.at[real, found]]
    #         axs[0, count].scatter(depth_matched, snr_matched, c=to_plot, label="matched")
    #         axs[0, count].scatter(depth_missed, snr_missed, c=np.zeros(len(snr_missed)), label="missed")
    #         axs[0, count].set_xlabel("snr")
    #         axs[0, count].set_ylabel(metric)
    #         label = self.cases[key]["label"]
    #         axs[0, count].set_title(label)
    #         axs[0, count].legend()

    def plot_unit_losses(self, before, after, metric="agreement", figsize=None):

        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=figsize)

        for count, k in enumerate(("accuracy", "recall", "precision")):

            ax = axs[count]

            label = self.cases[after]["label"]

            positions = self.get_result(before)["merging"].sorting1.get_property("gt_unit_locations")

            analyzer = self.get_sorting_analyzer(before)
            metrics_before = analyzer.get_extension("quality_metrics").get_data()
            x = metrics_before["snr"].values

            y_before = self.get_result(before)["merging"].get_performance()[k].values
            y_after = self.get_result(after)["merging"].get_performance()[k].values
            if count < 2:
                ax.set_xticks([], [])
            elif count == 2:
                ax.set_xlabel("depth (um)")
            im = ax.scatter(positions[:, 1], x, c=(y_after - y_before), marker=".", s=50, cmap="copper")
            fig.colorbar(im, ax=ax)
            ax.set_title(k)
            ax.set_ylabel("snr")

    # def plot_comparison_clustering(
    #     self,
    #     case_keys=None,
    #     performance_names=["accuracy", "recall", "precision"],
    #     colors=["g", "b", "r"],
    #     ylim=(-0.1, 1.1),
    #     figsize=None,
    # ):

    #     if case_keys is None:
    #         case_keys = list(self.cases.keys())

    #     num_methods = len(case_keys)
    #     fig, axs = plt.subplots(ncols=num_methods, nrows=num_methods, figsize=(10, 10))
    #     for i, key1 in enumerate(case_keys):
    #         for j, key2 in enumerate(case_keys):
    #             if len(axs.shape) > 1:
    #                 ax = axs[i, j]
    #             else:
    #                 ax = axs[j]
    #             comp1 = self.get_result(key1)["gt_comparison"]
    #             comp2 = self.get_result(key2)["gt_comparison"]
    #             if i <= j:
    #                 for performance, color in zip(performance_names, colors):
    #                     perf1 = comp1.get_performance()[performance]
    #                     perf2 = comp2.get_performance()[performance]
    #                     ax.plot(perf2, perf1, ".", label=performance, color=color)

    #                 ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    #                 ax.set_ylim(ylim)
    #                 ax.set_xlim(ylim)
    #                 ax.spines[["right", "top"]].set_visible(False)
    #                 ax.set_aspect("equal")

    #                 label1 = self.cases[key1]["label"]
    #                 label2 = self.cases[key2]["label"]
    #                 if j == i:
    #                     ax.set_ylabel(f"{label1}")
    #                 else:
    #                     ax.set_yticks([])
    #                 if i == j:
    #                     ax.set_xlabel(f"{label2}")
    #                 else:
    #                     ax.set_xticks([])
    #                 if i == num_methods - 1 and j == num_methods - 1:
    #                     patches = []
    #                     for color, name in zip(colors, performance_names):
    #                         patches.append(mpatches.Patch(color=color, label=name))
    #                     ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    #             else:
    #                 ax.spines["bottom"].set_visible(False)
    #                 ax.spines["left"].set_visible(False)
    #                 ax.spines["top"].set_visible(False)
    #                 ax.spines["right"].set_visible(False)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #     plt.tight_layout(h_pad=0, w_pad=0)
