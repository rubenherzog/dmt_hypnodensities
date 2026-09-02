"""Tests for reusable statistics and plotting functions."""

import unittest

import numpy as np
import pandas as pd

from dmt_hypnodensities.plots import (
    plot_condition_change_violins,
    plot_electrode_variance_violins,
    plot_entropy_distribution,
    plot_hypnodensity_condition_violins,
    plot_paired_condition_changes,
    plot_ranked_stage_features,
    plot_stage_feature_correlation_heatmap,
    plot_stage_feature_scatter,
    plot_stager_correlation_heatmap,
)
from dmt_hypnodensities.stats import (
    add_hypnodensity_entropy,
    adjust_pvalues,
    fit_mixed_models,
    paired_condition_wilcoxon,
    pairwise_stager_correlations,
    prepare_epoch_cohen_d,
    prepare_treatment_effects,
    prepare_within_condition_changes,
    stage_feature_effect_correlations,
)


def _hypnodensities() -> pd.DataFrame:
    rows = []
    for stager, wake_values in (("a", [0.1, 0.3, 0.7]), ("b", [0.2, 0.4, 0.8])):
        for epoch, wake in enumerate(wake_values):
            remainder = (1 - wake) / 4
            rows.append(
                {
                    "recording_id": "R",
                    "block_id": "B",
                    "epoch": epoch,
                    "channel_set": "E6",
                    "stager": stager,
                    "prob_W": wake,
                    "prob_N1": remainder,
                    "prob_N2": remainder,
                    "prob_N3": remainder,
                    "prob_R": remainder,
                }
            )
    return pd.DataFrame(rows)


def _paired_experiment() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for subject_index in range(12):
        subject = f"P{subject_index:02d}"
        for electrode_index, electrode in enumerate(("E6", "E8")):
            baseline = 0.2 + 0.002 * subject_index + 0.003 * electrode_index
            for condition in ("placebo", "DMT"):
                change = (
                    0.04 + 0.001 * subject_index + 0.001 * electrode_index
                    if condition == "placebo"
                    else 0.14 + 0.002 * subject_index - 0.001 * electrode_index
                )
                for label, center in (("before", baseline), ("after", baseline + change)):
                    for epoch in range(3):
                        wake = center + rng.normal(0, 0.0001)
                        remainder = 1 - wake
                        rows.append(
                            {
                                "subject": subject,
                                "condition": condition,
                                "electrode": electrode,
                                "stager": "yasa",
                                "experimental_label": label,
                                "intensity_label": "high" if subject_index < 6 else "low",
                                "epoch": epoch,
                                "prob_W": wake,
                                "prob_N1": remainder * 0.1,
                                "prob_N2": remainder * 0.2,
                                "prob_N3": remainder * 0.3,
                                "prob_R": remainder * 0.4,
                            }
                        )
    return pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)


class StatisticsAndPlotsTests(unittest.TestCase):
    def test_entropy_and_pairwise_correlations(self) -> None:
        table = add_hypnodensity_entropy(_hypnodensities())
        self.assertTrue(np.isfinite(table["entropy"]).all())

        correlations = pairwise_stager_correlations(table)

        wake = correlations.loc[correlations["metric"].eq("W")].iloc[0]
        self.assertEqual(wake["n"], 3)
        self.assertAlmostEqual(wake["correlation"], 1.0)
        self.assertIn("p_fdr_bh", correlations)

    def test_benjamini_hochberg_is_monotonic(self) -> None:
        adjusted = adjust_pvalues([0.01, 0.04, 0.03], method="fdr_bh")
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

    def test_wilcoxon_pairs_subjects_by_key_after_averaging_electrodes(self) -> None:
        table = _paired_experiment()
        results = paired_condition_wilcoxon(table, value_columns=("prob_W",))

        after = results.loc[results["experimental_label"].eq("after")].iloc[0]
        self.assertEqual(after["n_pairs"], 12)
        self.assertGreater(after["median_difference"], 0.09)
        self.assertGreater(after["rank_biserial"], 0.99)

        incomplete = table.loc[
            ~(
                table["subject"].eq("P00")
                & table["condition"].eq("DMT")
                & table["experimental_label"].eq("after")
            )
        ]
        incomplete_results = paired_condition_wilcoxon(incomplete, value_columns=("prob_W",))
        incomplete_after = incomplete_results.loc[
            incomplete_results["experimental_label"].eq("after")
        ].iloc[0]
        self.assertEqual(incomplete_after["n_pairs"], 11)

    def test_change_and_treatment_effect_directions_are_explicit(self) -> None:
        changes = prepare_within_condition_changes(
            _paired_experiment(),
            value_columns=("prob_W",),
            contrasts={"before_to_after": ("before", "after")},
            delta_types=("abs", "rel", "logit"),
        )
        row = changes.loc[
            changes["subject"].eq("P00")
            & changes["condition"].astype(str).eq("placebo")
            & changes["electrode"].eq("E6")
        ].iloc[0]
        self.assertAlmostEqual(row["prob_W__delta_abs"], 0.04, places=3)
        self.assertGreater(row["prob_W__delta_logit"], 0)

        effects = prepare_treatment_effects(
            changes,
            value_columns=("prob_W",),
            delta_types=("abs",),
        )
        self.assertTrue(effects["effect_direction"].eq("DMT_minus_placebo").all())
        self.assertAlmostEqual(effects["prob_W__effect"].median(), 0.105, places=2)

        effect_sizes = prepare_epoch_cohen_d(
            _paired_experiment(),
            value_columns=("prob_W",),
            contrasts={"before_to_after": ("before", "after")},
        )
        self.assertTrue(effect_sizes["prob_W__cohen_d"].gt(0).all())

    def test_mixed_model_recovers_condition_change(self) -> None:
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            self.skipTest("The optional statsmodels dependency is not installed.")
        changes = prepare_within_condition_changes(
            _paired_experiment(),
            value_columns=("prob_W",),
            contrasts={"before_to_after": ("before", "after")},
            delta_types=("abs",),
        )
        results = fit_mixed_models(
            changes,
            outcomes=("prob_W__delta_abs",),
            fixed_effects="condition",
            variance_components={"electrode": "0 + C(electrode)"},
            min_observations=20,
        )

        condition = results.loc[results["term"].str.contains("condition")].iloc[0]
        self.assertIn(condition["status"], {"ok", "not_converged"})
        self.assertAlmostEqual(condition["coefficient"], 0.105, places=2)

    def test_stage_feature_effect_correlations_preserve_pairwise_n(self) -> None:
        effects = pd.DataFrame(
            {
                "stager": ["gssc"] * 5,
                "contrast": ["before_to_after"] * 5,
                "delta_type": ["abs"] * 5,
                "prob_R__effect": [1.0, 2.0, 3.0, 4.0, 5.0],
                "bp_alpha__effect": [2.0, 4.0, 6.0, 8.0, np.nan],
            }
        )
        results = stage_feature_effect_correlations(
            effects,
            stage_columns=("prob_R__effect",),
            feature_columns=("bp_alpha__effect",),
            min_observations=3,
        )

        self.assertEqual(set(results["method"]), {"pearson", "spearman"})
        np.testing.assert_allclose(results["correlation"], 1.0)
        self.assertTrue(results["n_used"].eq(4).all())
        self.assertNotIn("p_value", results)

    def test_plot_functions_return_matplotlib_objects(self) -> None:
        try:
            import matplotlib
        except ImportError:
            self.skipTest("The optional matplotlib dependency is not installed.")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        entropy = add_hypnodensity_entropy(_hypnodensities())
        correlations = pairwise_stager_correlations(entropy)
        entropy_figure, _ = plot_entropy_distribution(entropy)
        heatmap_figure, _ = plot_stager_correlation_heatmap(correlations)
        plt.close(entropy_figure)
        plt.close(heatmap_figure)

    def test_notebook_hypnodensity_plot_layouts(self) -> None:
        try:
            import matplotlib
            import seaborn  # noqa: F401
        except ImportError:
            self.skipTest("The optional plotting dependencies are not installed.")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        table = _paired_experiment()
        original = table.copy(deep=True)
        wilcoxon = paired_condition_wilcoxon(table)
        figure, axes = plot_hypnodensity_condition_violins(
            table,
            wilcoxon_results=wilcoxon,
            labels=("before", "after"),
        )
        self.assertEqual(axes.shape, (1, 2))
        plt.close(figure)

        figure, axes = plot_hypnodensity_condition_violins(
            table,
            labels=("before", "after"),
            stratify_by="intensity_label",
        )
        self.assertEqual(axes.shape, (2, 2))
        plt.close(figure)

        figure, axes = plot_electrode_variance_violins(
            table,
            labels=("before", "after"),
        )
        self.assertEqual(len(axes), 2)
        plt.close(figure)
        pd.testing.assert_frame_equal(table, original)

    def test_notebook_change_and_feature_plot_layouts(self) -> None:
        try:
            import matplotlib
            import seaborn  # noqa: F401
        except ImportError:
            self.skipTest("The optional plotting dependencies are not installed.")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        table = _paired_experiment()
        probability_columns = tuple(f"prob_{stage}" for stage in ("W", "N1", "N2", "N3", "R"))
        changes = prepare_within_condition_changes(
            table,
            value_columns=probability_columns,
            contrasts={"before_to_after": ("before", "after")},
            delta_types=("abs",),
        )
        figure, axes = plot_paired_condition_changes(
            changes,
            contrasts=("before_to_after",),
        )
        self.assertEqual(axes.shape, (1, 5))
        plt.close(figure)

        effect_sizes = prepare_epoch_cohen_d(
            table,
            contrasts={"before_to_after": ("before", "after")},
        )
        figure, axes = plot_condition_change_violins(
            effect_sizes,
            contrasts=("before_to_after",),
        )
        self.assertEqual(len(axes), 1)
        plt.close(figure)
        figure, axes = plot_paired_condition_changes(
            effect_sizes,
            contrasts=("before_to_after",),
            value_template="prob_{stage}__cohen_d",
            ylabel="Cohen's d",
        )
        self.assertEqual(axes.shape, (1, 5))
        plt.close(figure)

        correlations = pd.DataFrame(
            [
                {
                    "stager": "gssc",
                    "contrast": "before_to_after",
                    "delta_type": "abs",
                    "method": "spearman",
                    "stage": stage,
                    "feature": feature,
                    "correlation": correlation,
                }
                for stage, correlation in (("W", -0.4), ("R", 0.7))
                for feature in ("bp_alpha", "specparam_exponent")
            ]
        )
        figure, axis = plot_stage_feature_correlation_heatmap(correlations)
        self.assertEqual(axis.get_ylabel(), "Sleep stage")
        plt.close(figure)
        figure, axis = plot_ranked_stage_features(correlations, stage="R")
        self.assertEqual(axis.get_xlim(), (-1.0, 1.0))
        plt.close(figure)

        joined = table.assign(bp_alpha=np.linspace(0, 1, len(table)))
        figure, axis = plot_stage_feature_scatter(joined, stage="R", feature="bp_alpha")
        self.assertEqual(axis.get_xlabel(), "R hypnodensity")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
