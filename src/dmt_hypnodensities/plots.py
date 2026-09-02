"""Small plotting API for already-computed analysis tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def save_figure(
    figure,
    output_stem: Path | str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Save one computed figure in the requested publication and preview formats."""

    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = stem.with_suffix(f".{extension.lstrip('.')}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return tuple(written)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Plotting requires matplotlib.") from error
    return plt


def _seaborn():
    try:
        import seaborn as sns
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Notebook-style plots require seaborn.") from error
    return sns


def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(table))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _select_one(table: pd.DataFrame, column: str, value: str | None) -> tuple[pd.DataFrame, str]:
    _require_columns(table, (column,), "Plot table")
    available = tuple(dict.fromkeys(table[column].dropna().astype(str)))
    if value is None:
        if len(available) != 1:
            raise ValueError(f"Select one {column}; available values are {available}.")
        value = available[0]
    if value not in available:
        raise ValueError(f"Unknown {column} {value!r}; available values are {available}.")
    return table.loc[table[column].astype(str).eq(value)].copy(), value


def _significance_symbol(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _subject_hypnodensity_long(
    table: pd.DataFrame,
    stages: Sequence[str],
    label_column: str,
    extra_groups: Sequence[str] = (),
) -> pd.DataFrame:
    probabilities = [f"prob_{stage}" for stage in stages]
    groups = ["subject", "condition", label_column, "stager", *extra_groups]
    _require_columns(table, (*groups, *probabilities), "Hypnodensity table")
    subject_means = (
        table.groupby(groups, observed=True, dropna=False)[probabilities].mean().reset_index()
    )
    return subject_means.melt(
        id_vars=groups,
        value_vars=probabilities,
        var_name="stage",
        value_name="probability",
    ).assign(stage=lambda frame: frame["stage"].str.removeprefix("prob_"))


def plot_hypnodensity_trajectories(
    hypnodensities: pd.DataFrame,
    stages: Sequence[str] = ("W", "N1", "N2", "N3", "R"),
    figsize: tuple[float, float] = (12, 10),
):
    """Plot mean epoch trajectories by stager for a pre-filtered recording."""

    required = {"epoch", "stager", *(f"prob_{stage}" for stage in stages)}
    missing = sorted(required - set(hypnodensities))
    if missing:
        raise ValueError(f"Hypnodensity table is missing columns: {missing}")
    plt = _pyplot()
    figure, axes = plt.subplots(len(stages), 1, figsize=figsize, sharex=True, squeeze=False)
    for axis, stage in zip(axes[:, 0], stages):
        for stager, group in hypnodensities.groupby("stager", sort=False):
            trajectory = group.groupby("epoch")[f"prob_{stage}"].mean().sort_index()
            axis.plot(trajectory.index, trajectory.values, label=str(stager))
        axis.set_ylabel(stage)
        axis.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, ncol=max(1, hypnodensities["stager"].nunique()))
    axes[-1, 0].set_xlabel("Epoch")
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_hypnodensity_condition_violins(
    hypnodensities: pd.DataFrame,
    wilcoxon_results: pd.DataFrame | None = None,
    stager: str | None = None,
    stages: Sequence[str] = ("N1", "N2", "N3", "R", "W"),
    labels: Sequence[str] = ("before", "after", "late"),
    label_column: str = "experimental_label",
    stratify_by: str | None = None,
    control: str = "placebo",
    treatment: str = "DMT",
    adjusted_p_column: str = "p_fdr_bh",
    figsize: tuple[float, float] | None = None,
):
    """Reproduce condition-split hypnodensity violins using subject-level means.

    ``stratify_by="intensity_label"`` produces the 2 x 3 intensity layout from the
    notebooks. Optional Wilcoxon results annotate FDR-adjusted significance.
    """

    groups = (stratify_by,) if stratify_by else ()
    selected, selected_stager = _select_one(hypnodensities, "stager", stager)
    long = _subject_hypnodensity_long(selected, stages, label_column, groups)
    strata = tuple(dict.fromkeys(long[stratify_by].dropna())) if stratify_by else (None,)
    n_rows, n_columns = len(strata), len(labels)
    if not n_rows:
        raise ValueError(f"No finite values are available for {stratify_by!r}.")
    plt = _pyplot()
    sns = _seaborn()
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figsize or (6 * n_columns, 5 * n_rows),
        sharey=True,
        squeeze=False,
    )
    palette = {control: "#777777", treatment: "#D62728"}
    for row, stratum in enumerate(strata):
        for column, label in enumerate(labels):
            axis = axes[row, column]
            subset = long.loc[long[label_column].eq(label)]
            if stratify_by:
                subset = subset.loc[subset[stratify_by].eq(stratum)]
            sns.violinplot(
                data=subset,
                x="stage",
                y="probability",
                hue="condition",
                order=list(stages),
                hue_order=[control, treatment],
                split=True,
                inner="quartile",
                palette=palette,
                cut=0,
                density_norm="width",
                gap=0.1,
                fill=False,
                linewidth=2,
                ax=axis,
            )
            axis.set_ylim(0, 1.08)
            axis.set_title(str(label).capitalize() if row == 0 else "")
            axis.set_xlabel("Sleep stage")
            ylabel = "Hypnodensity probability"
            if stratify_by and column == 0:
                ylabel += f"\n{stratify_by}={stratum}"
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(axis="y", linestyle="--", alpha=0.5)
            legend = axis.get_legend()
            if legend is not None and (row, column) != (0, 0):
                legend.remove()
            elif legend is not None:
                legend.set_title(None)

            if wilcoxon_results is not None:
                _require_columns(
                    wilcoxon_results,
                    (label_column, "metric", adjusted_p_column),
                    "Wilcoxon table",
                )
                stats = wilcoxon_results.loc[
                    wilcoxon_results[label_column].eq(label)
                    & wilcoxon_results["metric"].isin(stages)
                ]
                if "stager" in stats:
                    stats = stats.loc[stats["stager"].astype(str).eq(selected_stager)]
                if stratify_by and stratify_by in stats:
                    stats = stats.loc[stats[stratify_by].eq(stratum)]
                p_values = stats.set_index("metric")[adjusted_p_column]
                for position, stage in enumerate(stages):
                    symbol = _significance_symbol(float(p_values.get(stage, np.nan)))
                    if symbol:
                        axis.text(position, 1.015, symbol, ha="center", va="bottom", fontsize=13)
    figure.suptitle(f"Hypnodensities by condition — {selected_stager}", y=1.01)
    figure.tight_layout()
    return figure, axes


def plot_electrode_variance_violins(
    hypnodensities: pd.DataFrame,
    stager: str | None = None,
    stages: Sequence[str] = ("N1", "N2", "N3", "R", "W"),
    labels: Sequence[str] = ("before", "after", "late"),
    label_column: str = "experimental_label",
    control: str = "placebo",
    treatment: str = "DMT",
    figsize: tuple[float, float] | None = None,
):
    """Plot subject-level variance across electrode means, as in the notebooks."""

    selected, selected_stager = _select_one(hypnodensities, "stager", stager)
    probabilities = [f"prob_{stage}" for stage in stages]
    groups = ["subject", "condition", label_column, "electrode", "stager"]
    _require_columns(selected, (*groups, *probabilities), "Hypnodensity table")
    electrode_means = (
        selected.groupby(groups, observed=True, dropna=False)[probabilities].mean().reset_index()
    )
    variances = (
        electrode_means.groupby(
            ["subject", "condition", label_column, "stager"], observed=True, dropna=False
        )[probabilities]
        .var()
        .reset_index()
        .melt(
            id_vars=["subject", "condition", label_column, "stager"],
            value_vars=probabilities,
            var_name="stage",
            value_name="variance",
        )
    )
    variances["stage"] = variances["stage"].str.removeprefix("prob_")
    plt = _pyplot()
    sns = _seaborn()
    figure, axes = plt.subplots(
        1,
        len(labels),
        figsize=figsize or (6 * len(labels), 5),
        sharey=True,
        squeeze=False,
    )
    palette = {control: "#777777", treatment: "#D62728"}
    for column, label in enumerate(labels):
        axis = axes[0, column]
        subset = variances.loc[variances[label_column].eq(label)]
        sns.violinplot(
            data=subset,
            x="stage",
            y="variance",
            hue="condition",
            order=list(stages),
            hue_order=[control, treatment],
            split=True,
            inner="quartile",
            palette=palette,
            cut=0,
            density_norm="width",
            gap=0.1,
            fill=False,
            linewidth=2,
            ax=axis,
        )
        axis.set_title(str(label).capitalize())
        axis.set_xlabel("Sleep stage")
        axis.set_ylabel("Variance across electrodes" if column == 0 else "")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        legend = axis.get_legend()
        if legend is not None and column != 0:
            legend.remove()
        elif legend is not None:
            legend.set_title(None)
    figure.suptitle(f"Electrode variance of mean hypnodensity — {selected_stager}", y=1.01)
    figure.tight_layout()
    return figure, axes[0]


def plot_condition_change_violins(
    changes: pd.DataFrame,
    stager: str | None = None,
    stages: Sequence[str] = ("N1", "N2", "N3", "R", "W"),
    contrasts: Sequence[str] = ("before_to_after", "before_to_late"),
    value_template: str = "prob_{stage}__cohen_d",
    control: str = "placebo",
    treatment: str = "DMT",
    ylabel: str = "Cohen's d (follow-up − baseline)",
    figsize: tuple[float, float] | None = None,
):
    """Plot condition-split change violins in the notebook's contrast layout."""

    selected, selected_stager = _select_one(changes, "stager", stager)
    value_columns = [value_template.format(stage=stage) for stage in stages]
    groups = ["subject", "condition", "contrast", "stager"]
    _require_columns(selected, (*groups, *value_columns), "Change table")
    subject_means = (
        selected.groupby(groups, observed=True, dropna=False)[value_columns].mean().reset_index()
    )
    long = subject_means.melt(
        id_vars=groups,
        value_vars=value_columns,
        var_name="value_column",
        value_name="change",
    )
    column_to_stage = dict(zip(value_columns, stages))
    long["stage"] = long["value_column"].map(column_to_stage)
    plt = _pyplot()
    sns = _seaborn()
    figure, axes = plt.subplots(
        1,
        len(contrasts),
        figsize=figsize or (7 * len(contrasts), 5),
        sharey=True,
        squeeze=False,
    )
    palette = {control: "#777777", treatment: "#D62728"}
    for column, contrast in enumerate(contrasts):
        axis = axes[0, column]
        subset = long.loc[long["contrast"].eq(contrast)]
        axis.axhline(0, color="black", linewidth=1)
        sns.violinplot(
            data=subset,
            x="stage",
            y="change",
            hue="condition",
            order=list(stages),
            hue_order=[control, treatment],
            split=True,
            inner="quartile",
            palette=palette,
            cut=0,
            gap=0.1,
            fill=False,
            linewidth=2,
            ax=axis,
        )
        axis.set_title(contrast.replace("_", " ").capitalize())
        axis.set_xlabel("Sleep stage")
        axis.set_ylabel(ylabel if column == 0 else "")
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        legend = axis.get_legend()
        if legend is not None and column != 0:
            legend.remove()
        elif legend is not None:
            legend.set_title(None)
    figure.suptitle(f"Within-condition effect sizes — {selected_stager}", y=1.01)
    figure.tight_layout()
    return figure, axes[0]


def plot_paired_condition_changes(
    changes: pd.DataFrame,
    stager: str | None = None,
    stages: Sequence[str] = ("N1", "N2", "N3", "R", "W"),
    contrasts: Sequence[str] = ("before_to_after", "before_to_late"),
    value_template: str = "prob_{stage}__delta_abs",
    control: str = "placebo",
    treatment: str = "DMT",
    color_by: str | None = None,
    random_seed: int = 7,
    ylabel: str = "Change (follow-up − baseline)",
    figsize: tuple[float, float] | None = None,
):
    """Plot paired subject changes in the 2 x 5 layout used by the notebooks."""

    selected, selected_stager = _select_one(changes, "stager", stager)
    value_columns = [value_template.format(stage=stage) for stage in stages]
    groups = ["subject", "condition", "contrast", "stager"]
    if color_by:
        groups.append(color_by)
    _require_columns(selected, (*groups, *value_columns), "Change table")
    subject_means = (
        selected.groupby(groups, observed=True, dropna=False)[value_columns].mean().reset_index()
    )
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(contrasts),
        len(stages),
        figsize=figsize or (3.6 * len(stages), 3.4 * len(contrasts)),
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(random_seed)
    subjects = tuple(sorted(subject_means["subject"].astype(str).unique()))
    offsets = dict(zip(subjects, rng.uniform(-0.08, 0.08, size=len(subjects))))
    if color_by:
        categories = tuple(dict.fromkeys(subject_means[color_by].dropna().astype(str)))
        colors = dict(zip(categories, plt.get_cmap("tab10").colors))
    else:
        colors = {}
    for row, contrast in enumerate(contrasts):
        for column, stage in enumerate(stages):
            axis = axes[row, column]
            value = value_template.format(stage=stage)
            subset = subject_means.loc[subject_means["contrast"].eq(contrast)]
            for subject, subject_data in subset.groupby("subject", observed=True, sort=False):
                paired = subject_data.set_index("condition")
                if control not in paired.index or treatment not in paired.index:
                    continue
                values = [float(paired.loc[control, value]), float(paired.loc[treatment, value])]
                if not np.isfinite(values).all():
                    continue
                offset = offsets[str(subject)]
                color = (
                    colors.get(str(paired.iloc[0][color_by]), "#555555") if color_by else "#666666"
                )
                axis.plot([offset, 1 + offset], values, color=color, alpha=0.5, linewidth=1)
                axis.scatter(
                    [offset, 1 + offset],
                    values,
                    c=["#777777", "#D62728"] if not color_by else [color, color],
                    s=24,
                    zorder=3,
                )
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_xticks([0, 1], [control.capitalize(), treatment])
            axis.set_title(f"{stage} — {contrast.replace('_', ' ')}")
            axis.set_xlabel("")
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(axis="y", linestyle="--", alpha=0.4)
    if color_by and colors:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
            for label, color in colors.items()
        ]
        figure.legend(handles=handles, title=color_by, loc="upper right")
    figure.suptitle(f"Paired condition changes — {selected_stager}", y=1.01)
    figure.tight_layout()
    return figure, axes


def plot_stager_correlation_heatmap(
    correlations: pd.DataFrame,
    value: str = "correlation",
    figsize: tuple[float, float] = (8, 4),
):
    """Plot pairwise stager correlations with metrics as columns."""

    required = {"stager_a", "stager_b", "metric", value}
    missing = sorted(required - set(correlations))
    if missing:
        raise ValueError(f"Correlation table is missing columns: {missing}")
    table = correlations.copy()
    table["comparison"] = table["stager_a"] + " vs " + table["stager_b"]
    matrix = table.pivot_table(index="comparison", columns="metric", values=value)
    preferred = [stage for stage in ("W", "N1", "N2", "N3", "R", "entropy") if stage in matrix]
    matrix = matrix.reindex(columns=preferred)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            number = matrix.iat[row, column]
            if np.isfinite(number):
                axis.text(column, row, f"{number:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label=value)
    figure.tight_layout()
    return figure, axis


def plot_stage_feature_correlation_heatmap(
    correlations: pd.DataFrame,
    stager: str | None = None,
    contrast: str | None = None,
    delta_type: str | None = None,
    method: str | None = None,
    stages: Sequence[str] = ("W", "N1", "N2", "N3", "R"),
    max_feature_ticks: int = 60,
    figsize: tuple[float, float] | None = None,
):
    """Plot the stage x feature effect-correlation heatmap from the joint notebook."""

    selected = correlations.copy()
    labels = {}
    for column, value in (
        ("stager", stager),
        ("contrast", contrast),
        ("delta_type", delta_type),
        ("method", method),
    ):
        selected, labels[column] = _select_one(selected, column, value)
    _require_columns(selected, ("stage", "feature", "correlation"), "Correlation table")
    if selected.duplicated(["stage", "feature"]).any():
        raise ValueError("Stage-feature correlations are not unique after filtering.")
    matrix = selected.pivot(index="stage", columns="feature", values="correlation")
    matrix = matrix.reindex([stage for stage in stages if stage in matrix.index])
    plt = _pyplot()
    width = max(10.0, 0.28 * len(matrix.columns))
    figure, axis = plt.subplots(figsize=figsize or (width, 5))
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    tick_step = max(1, int(np.ceil(len(matrix.columns) / max_feature_ticks)))
    positions = np.arange(0, len(matrix.columns), tick_step)
    axis.set_xticks(positions, [matrix.columns[index] for index in positions], rotation=90)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_xlabel("Feature")
    axis.set_ylabel("Sleep stage")
    axis.set_title(
        f"Stage x feature correlation — {labels['stager']} — {labels['contrast']} — "
        f"{labels['delta_type']} — {labels['method']}"
    )
    figure.colorbar(image, ax=axis, label="Correlation r")
    figure.tight_layout()
    return figure, axis


def plot_ranked_stage_features(
    correlations: pd.DataFrame,
    stage: str = "R",
    stager: str | None = None,
    contrast: str | None = None,
    delta_type: str | None = None,
    method: str | None = None,
    top_n: int = 20,
    figsize: tuple[float, float] | None = None,
):
    """Plot the strongest absolute feature correlations for one sleep stage."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")
    selected = correlations.loc[correlations["stage"].eq(stage)].copy()
    labels = {}
    for column, value in (
        ("stager", stager),
        ("contrast", contrast),
        ("delta_type", delta_type),
        ("method", method),
    ):
        selected, labels[column] = _select_one(selected, column, value)
    _require_columns(selected, ("feature", "correlation"), "Correlation table")
    ranked = selected.dropna(subset=["correlation"]).copy()
    ranked["absolute_correlation"] = ranked["correlation"].abs()
    ranked = ranked.nlargest(top_n, "absolute_correlation").sort_values("correlation")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize or (10, max(4, 0.32 * len(ranked))))
    colors = plt.get_cmap("coolwarm")((ranked["correlation"].to_numpy() + 1) / 2)
    axis.barh(ranked["feature"], ranked["correlation"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Correlation r")
    axis.set_ylabel("Feature")
    axis.set_title(
        f"{stage} feature ranking — {labels['stager']} — {labels['contrast']} — "
        f"{labels['delta_type']} — {labels['method']}"
    )
    figure.tight_layout()
    return figure, axis


def plot_stage_feature_scatter(
    table: pd.DataFrame,
    stage: str,
    feature: str,
    condition_column: str = "condition",
    figsize: tuple[float, float] = (7, 5),
):
    """Plot the condition-colored stage/feature scatter used for exploration."""

    probability = f"prob_{stage}"
    _require_columns(table, (probability, feature, condition_column), "Epoch analysis table")
    plot_data = table[[probability, feature, condition_column]].dropna()
    plt = _pyplot()
    sns = _seaborn()
    figure, axis = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=plot_data,
        x=probability,
        y=feature,
        hue=condition_column,
        palette={"placebo": "#777777", "DMT": "#D62728"},
        alpha=0.55,
        s=28,
        ax=axis,
    )
    axis.set_xlabel(f"{stage} hypnodensity")
    axis.set_ylabel(feature)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure, axis


def plot_entropy_distribution(
    table: pd.DataFrame,
    entropy_column: str = "entropy",
    figsize: tuple[float, float] = (7, 4),
):
    """Plot a boxplot of precomputed entropy values by stager."""

    if entropy_column not in table or "stager" not in table:
        raise ValueError("Entropy plots require 'stager' and the entropy column.")
    stagers = list(dict.fromkeys(table["stager"].astype(str)))
    values = [
        table.loc[table["stager"].astype(str).eq(stager), entropy_column].dropna()
        for stager in stagers
    ]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize)
    axis.boxplot(values, tick_labels=stagers, showfliers=False)
    axis.set_ylabel("Shannon entropy")
    axis.set_xlabel("Stager")
    figure.tight_layout()
    return figure, axis


def plot_feature_by_group(
    table: pd.DataFrame,
    feature: str,
    group: str = "condition",
    figsize: tuple[float, float] = (7, 4),
):
    """Plot group means and standard errors for any numeric feature column."""

    if feature not in table or group not in table:
        raise ValueError(f"Table must contain {feature!r} and {group!r}.")
    summary = table.groupby(group, dropna=False)[feature].agg(["mean", "sem"])
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize)
    positions = np.arange(len(summary))
    axis.errorbar(positions, summary["mean"], yerr=summary["sem"], fmt="o", capsize=4)
    axis.set_xticks(positions, summary.index.astype(str))
    axis.set_xlabel(group)
    axis.set_ylabel(feature)
    figure.tight_layout()
    return figure, axis
