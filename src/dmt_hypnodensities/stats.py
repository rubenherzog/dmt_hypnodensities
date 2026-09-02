"""Reusable statistics for feature and hypnodensity tables."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

STAGES = ("W", "N1", "N2", "N3", "R")
PROBABILITY_COLUMNS = tuple(f"prob_{stage}" for stage in STAGES)


def _require_columns(table: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def add_hypnodensity_entropy(
    table: pd.DataFrame,
    normalized: bool = False,
    column: str = "entropy",
) -> pd.DataFrame:
    """Add Shannon entropy from W/N1/N2/N3/R probabilities."""

    missing = [name for name in PROBABILITY_COLUMNS if name not in table]
    if missing:
        raise ValueError(f"Hypnodensity table is missing probability columns: {missing}")
    output = table.copy()
    probabilities = output[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Hypnodensity probabilities must lie between zero and one.")
    finite_rows = np.isfinite(probabilities).all(axis=1)
    row_sums = probabilities.sum(axis=1)
    if np.any(finite_rows & ~np.isclose(row_sums, 1.0, atol=1e-5)):
        raise ValueError("Finite hypnodensity rows must sum to one.")
    entropy = np.full(len(output), np.nan)
    valid = finite_rows
    values = probabilities[valid]
    positive = values > 0
    terms = np.zeros_like(values)
    terms[positive] = values[positive] * np.log(values[positive])
    entropy[valid] = -terms.sum(axis=1)
    if normalized:
        entropy /= np.log(len(STAGES))
    output[column] = entropy
    return output


def adjust_pvalues(
    values: Sequence[float],
    method: str = "fdr_bh",
) -> np.ndarray:
    """Adjust finite p-values with Benjamini–Hochberg or Bonferroni."""

    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    finite = p_values[finite_indices]
    if not finite.size:
        return adjusted
    if method == "bonferroni":
        adjusted[finite_indices] = np.minimum(finite * finite.size, 1.0)
    elif method == "fdr_bh":
        order = np.argsort(finite)
        ranked = finite[order] * finite.size / np.arange(1, finite.size + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty_like(ranked)
        restored[order] = np.minimum(ranked, 1.0)
        adjusted[finite_indices] = restored
    else:
        raise ValueError(f"Unsupported p-value adjustment method: {method!r}.")
    return adjusted


def paired_condition_wilcoxon(
    table: pd.DataFrame,
    value_columns: Sequence[str] = PROBABILITY_COLUMNS,
    group_by: Sequence[str] = ("stager", "experimental_label"),
    subject_column: str = "subject",
    condition_column: str = "condition",
    treatment: str = "DMT",
    control: str = "placebo",
    p_adjust: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Compare paired subject means between treatment and control with Wilcoxon.

    Epochs and electrodes are averaged within each subject, condition and requested group
    before pairing. This matches the intended subject-level test in the notebooks while
    avoiding their positional pairing and electrode pseudoreplication.
    """

    try:
        from scipy.stats import rankdata, wilcoxon
    except ImportError as error:  # pragma: no cover - optional stats dependency
        raise ImportError("Wilcoxon tests require scipy.") from error
    values = tuple(value_columns)
    keys = (subject_column, condition_column, *group_by)
    _require_columns(table, (*keys, *values), "Analysis table")
    if treatment == control:
        raise ValueError("Treatment and control labels must differ.")

    means = (
        table.groupby(list(keys), observed=True, dropna=False)[list(values)].mean().reset_index()
    )
    rows = []
    grouper = group_by[0] if len(group_by) == 1 else list(group_by)
    grouped = [((), means)] if not group_by else means.groupby(grouper, observed=True, dropna=False)
    for group_values, subset in grouped:
        if group_by and not isinstance(group_values, tuple):
            group_values = (group_values,)
        metadata = dict(zip(group_by, group_values)) if group_by else {}
        for value in values:
            paired = subset.pivot(index=subject_column, columns=condition_column, values=value)
            if treatment not in paired or control not in paired:
                complete = pd.DataFrame(columns=[control, treatment])
            else:
                complete = paired[[control, treatment]].dropna()
            differences = complete[treatment].to_numpy() - complete[control].to_numpy()
            nonzero = differences[~np.isclose(differences, 0.0)]
            if nonzero.size:
                result = wilcoxon(
                    complete[treatment],
                    complete[control],
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                ranks = rankdata(np.abs(nonzero))
                positive = ranks[nonzero > 0].sum()
                negative = ranks[nonzero < 0].sum()
                rank_biserial = (positive - negative) / (positive + negative)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            elif len(complete):
                statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
            else:
                statistic, p_value, rank_biserial = np.nan, np.nan, np.nan
            rows.append(
                {
                    **metadata,
                    "metric": value.removeprefix("prob_"),
                    "contrast": f"{treatment}_minus_{control}",
                    "n_pairs": len(complete),
                    "n_nonzero_pairs": int(nonzero.size),
                    "control_mean": complete[control].mean() if len(complete) else np.nan,
                    "treatment_mean": complete[treatment].mean() if len(complete) else np.nan,
                    "median_difference": np.median(differences) if len(complete) else np.nan,
                    "statistic": statistic,
                    "rank_biserial": rank_biserial,
                    "p_value": p_value,
                }
            )
    output = pd.DataFrame(rows)
    if p_adjust is not None and not output.empty:
        output[f"p_{p_adjust}"] = adjust_pvalues(output["p_value"], method=p_adjust)
    return output


def prepare_within_condition_changes(
    table: pd.DataFrame,
    value_columns: Sequence[str],
    contrasts: Mapping[str, tuple[str, str]] | None = None,
    entity_columns: Sequence[str] = ("subject", "condition", "electrode", "stager"),
    label_column: str = "experimental_label",
    delta_types: Sequence[str] = ("abs", "rel"),
    relative_epsilon: float = 1e-6,
    logit_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Epoch-average variables and calculate explicit follow-up minus baseline changes."""

    contrasts = contrasts or {
        "before_to_after": ("before", "after"),
        "before_to_late": ("before", "late"),
    }
    values = tuple(value_columns)
    entities = tuple(entity_columns)
    requested = tuple(delta_types)
    unknown = sorted(set(requested) - {"abs", "rel", "logit"})
    if unknown:
        raise ValueError(f"Unsupported delta types: {unknown}")
    if relative_epsilon <= 0 or logit_epsilon <= 0 or logit_epsilon >= 0.5:
        raise ValueError("Epsilon values must be positive; logit_epsilon must be below 0.5.")
    _require_columns(table, (*entities, label_column, *values), "Analysis table")

    means = (
        table.groupby([*entities, label_column], observed=True, dropna=False)[list(values)]
        .mean()
        .reset_index()
    )
    frames = []
    for contrast, (baseline_label, followup_label) in contrasts.items():
        baseline = means.loc[means[label_column].eq(baseline_label), [*entities, *values]]
        followup = means.loc[means[label_column].eq(followup_label), [*entities, *values]]
        paired = baseline.merge(
            followup,
            on=list(entities),
            how="inner",
            suffixes=("__baseline", "__followup"),
            validate="one_to_one",
        )
        output = paired[list(entities)].copy()
        output["contrast"] = contrast
        output["baseline_label"] = baseline_label
        output["followup_label"] = followup_label
        for value in values:
            baseline_values = paired[f"{value}__baseline"].to_numpy(dtype=float)
            followup_values = paired[f"{value}__followup"].to_numpy(dtype=float)
            difference = followup_values - baseline_values
            if "abs" in requested:
                output[f"{value}__delta_abs"] = difference
            if "rel" in requested:
                sign = np.where(baseline_values < 0, -1.0, 1.0)
                denominator = sign * np.maximum(np.abs(baseline_values), relative_epsilon)
                output[f"{value}__delta_rel"] = difference / denominator
            if "logit" in requested:
                if np.any((baseline_values < 0) | (baseline_values > 1)) or np.any(
                    (followup_values < 0) | (followup_values > 1)
                ):
                    raise ValueError(f"Logit changes require {value!r} to lie in [0, 1].")
                baseline_clipped = np.clip(baseline_values, logit_epsilon, 1 - logit_epsilon)
                followup_clipped = np.clip(followup_values, logit_epsilon, 1 - logit_epsilon)
                output[f"{value}__delta_logit"] = np.log(
                    followup_clipped / (1 - followup_clipped)
                ) - np.log(baseline_clipped / (1 - baseline_clipped))
        frames.append(output)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "condition" in result:
        labels = list(dict.fromkeys(["placebo", "DMT", *result["condition"].astype(str)]))
        result["condition"] = pd.Categorical(result["condition"], categories=labels)
    return result


def prepare_epoch_cohen_d(
    table: pd.DataFrame,
    value_columns: Sequence[str] = PROBABILITY_COLUMNS,
    contrasts: Mapping[str, tuple[str, str]] | None = None,
    entity_columns: Sequence[str] = ("subject", "condition", "electrode", "stager"),
    label_column: str = "experimental_label",
) -> pd.DataFrame:
    """Calculate pooled-SD Cohen's d between epoch distributions within each entity.

    The sign is always follow-up minus baseline. This reproduces the descriptive effect
    sizes plotted in the notebooks; it is not used as an independent inferential test.
    """

    contrasts = contrasts or {
        "before_to_after": ("before", "after"),
        "before_to_late": ("before", "late"),
    }
    values = tuple(value_columns)
    entities = tuple(entity_columns)
    _require_columns(table, (*entities, label_column, *values), "Analysis table")
    rows = []
    for entity_values, subset in table.groupby(
        list(entities), observed=True, dropna=False, sort=False
    ):
        if not isinstance(entity_values, tuple):
            entity_values = (entity_values,)
        metadata = dict(zip(entities, entity_values))
        for contrast, (baseline_label, followup_label) in contrasts.items():
            baseline = subset.loc[subset[label_column].eq(baseline_label)]
            followup = subset.loc[subset[label_column].eq(followup_label)]
            output = {
                **metadata,
                "contrast": contrast,
                "baseline_label": baseline_label,
                "followup_label": followup_label,
            }
            for value in values:
                first = baseline[value].dropna().to_numpy(dtype=float)
                second = followup[value].dropna().to_numpy(dtype=float)
                degrees = len(first) + len(second) - 2
                if len(first) >= 2 and len(second) >= 2 and degrees > 0:
                    pooled_variance = (
                        (len(first) - 1) * np.var(first, ddof=1)
                        + (len(second) - 1) * np.var(second, ddof=1)
                    ) / degrees
                    pooled_sd = np.sqrt(pooled_variance)
                    effect_size = (
                        (np.mean(second) - np.mean(first)) / pooled_sd if pooled_sd > 0 else np.nan
                    )
                else:
                    effect_size = np.nan
                output[f"{value}__cohen_d"] = effect_size
            rows.append(output)
    return pd.DataFrame(rows)


def prepare_treatment_effects(
    changes: pd.DataFrame,
    value_columns: Sequence[str],
    delta_types: Sequence[str] = ("abs", "rel"),
    pairing_columns: Sequence[str] = ("subject", "electrode", "stager", "contrast"),
    condition_column: str = "condition",
    treatment: str = "DMT",
    control: str = "placebo",
) -> pd.DataFrame:
    """Calculate net effects as treatment change minus paired control change."""

    values = tuple(value_columns)
    pair_keys = tuple(pairing_columns)
    delta_columns = [f"{value}__delta_{kind}" for kind in delta_types for value in values]
    _require_columns(changes, (*pair_keys, condition_column, *delta_columns), "Change table")
    frames = []
    for kind in delta_types:
        columns = [f"{value}__delta_{kind}" for value in values]
        subset = changes[[*pair_keys, condition_column, *columns]]
        treated = subset.loc[subset[condition_column].astype(str).eq(treatment)].drop(
            columns=condition_column
        )
        controlled = subset.loc[subset[condition_column].astype(str).eq(control)].drop(
            columns=condition_column
        )
        paired = treated.merge(
            controlled,
            on=list(pair_keys),
            how="inner",
            suffixes=("__treatment", "__control"),
            validate="one_to_one",
        )
        output = paired[list(pair_keys)].copy()
        output["delta_type"] = kind
        output["effect_direction"] = f"{treatment}_minus_{control}"
        for value, delta_column in zip(values, columns):
            output[f"{value}__effect"] = (
                paired[f"{delta_column}__treatment"] - paired[f"{delta_column}__control"]
            )
        frames.append(output)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stage_feature_effect_correlations(
    effects: pd.DataFrame,
    stage_columns: Sequence[str],
    feature_columns: Sequence[str],
    group_by: Sequence[str] = ("stager", "contrast", "delta_type"),
    methods: Sequence[str] = ("spearman", "pearson"),
    min_observations: int = 30,
) -> pd.DataFrame:
    """Correlate aligned DMT-minus-placebo stage and feature effects.

    This is the exploratory correlation matrix from the joint GSSC/features notebook.
    It deliberately reports correlations and availability without inferential p-values;
    confirmatory feature associations belong in mixed-effects models.
    """

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as error:  # pragma: no cover - optional stats dependency
        raise ImportError("Effect correlations require scipy.") from error
    stages = tuple(stage_columns)
    features = tuple(feature_columns)
    groups = tuple(group_by)
    requested_methods = tuple(methods)
    unknown = sorted(set(requested_methods) - {"pearson", "spearman"})
    if unknown:
        raise ValueError(f"Unsupported correlation methods: {unknown}")
    if min_observations < 2:
        raise ValueError("min_observations must be at least two.")
    _require_columns(effects, (*groups, *stages, *features), "Treatment-effect table")

    grouped = [((), effects)]
    if groups:
        grouper = groups[0] if len(groups) == 1 else list(groups)
        grouped = list(effects.groupby(grouper, observed=True, dropna=False, sort=False))
    rows = []
    for group_values, subset in grouped:
        if groups and not isinstance(group_values, tuple):
            group_values = (group_values,)
        metadata = dict(zip(groups, group_values)) if groups else {}
        for method in requested_methods:
            for stage in stages:
                x = subset[stage].to_numpy(dtype=float)
                for feature in features:
                    y = subset[feature].to_numpy(dtype=float)
                    valid = np.isfinite(x) & np.isfinite(y)
                    n_used = int(valid.sum())
                    if n_used >= min_observations and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
                        result = (
                            pearsonr(x[valid], y[valid])
                            if method == "pearson"
                            else spearmanr(x[valid], y[valid])
                        )
                        correlation = float(result.statistic)
                    else:
                        correlation = np.nan
                    rows.append(
                        {
                            **metadata,
                            "method": method,
                            "stage": stage.removeprefix("prob_").removesuffix("__effect"),
                            "stage_column": stage,
                            "feature": feature.removesuffix("__effect"),
                            "feature_column": feature,
                            "correlation": correlation,
                            "n_used": n_used,
                            "n_total": len(subset),
                            "n_missing": len(subset) - n_used,
                            "min_n_passed": n_used >= min_observations,
                        }
                    )
    return pd.DataFrame(rows)


def fit_mixed_models(
    table: pd.DataFrame,
    outcomes: Sequence[str],
    fixed_effects: str,
    group_column: str = "subject",
    variance_components: Mapping[str, str] | None = None,
    stratify_by: Sequence[str] = ("stager",),
    re_formula: str = "1",
    reml: bool = False,
    method: str = "lbfgs",
    maxiter: int = 400,
    min_observations: int = 10,
    p_adjust: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Fit repeated MixedLM formulas and return a serializable coefficient table."""

    try:
        import statsmodels.formula.api as smf
    except ImportError as error:  # pragma: no cover - optional stats dependency
        raise ImportError("Linear mixed-effects models require statsmodels.") from error
    outcome_columns = tuple(outcomes)
    strata = tuple(stratify_by)
    variance = dict(variance_components or {})
    _require_columns(table, (*outcome_columns, group_column, *strata), "Mixed-model table")
    for component in variance:
        if component not in table:
            raise ValueError(f"Variance-component column {component!r} is absent.")
    if min_observations < 1:
        raise ValueError("min_observations must be positive.")

    grouped = [((), table)]
    if strata:
        grouper = strata[0] if len(strata) == 1 else list(strata)
        grouped = list(table.groupby(grouper, observed=True, dropna=False, sort=False))
    rows = []
    for group_values, subset in grouped:
        if strata and not isinstance(group_values, tuple):
            group_values = (group_values,)
        metadata = dict(zip(strata, group_values)) if strata else {}
        for outcome in outcome_columns:
            required_for_fit = [outcome, group_column, *variance]
            data = subset.dropna(subset=required_for_fit).copy()
            base = {
                **metadata,
                "outcome": outcome,
                "formula": f'Q("{outcome}") ~ {fixed_effects}',
                "n_used": len(data),
                "n_groups": data[group_column].nunique(),
            }
            if len(data) < min_observations:
                rows.append(
                    {
                        **base,
                        "term": "",
                        "status": "too_few_observations",
                        "coefficient": np.nan,
                        "std_error": np.nan,
                        "z_value": np.nan,
                        "p_value": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "converged": False,
                        "warning": "",
                        "error": "",
                    }
                )
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = smf.mixedlm(
                        base["formula"],
                        data=data,
                        groups=data[group_column],
                        vc_formula=variance or None,
                        re_formula=re_formula,
                    )
                    fit = model.fit(
                        reml=reml,
                        method=method,
                        maxiter=maxiter,
                        disp=False,
                    )
                confidence = fit.conf_int()
                warning_text = "; ".join(dict.fromkeys(str(item.message) for item in caught))
                for term in fit.fe_params.index:
                    rows.append(
                        {
                            **base,
                            "term": term,
                            "status": "ok" if fit.converged else "not_converged",
                            "coefficient": float(fit.fe_params[term]),
                            "std_error": float(fit.bse_fe[term]),
                            "z_value": float(fit.tvalues[term]),
                            "p_value": float(fit.pvalues[term]),
                            "ci95_low": float(confidence.loc[term, 0]),
                            "ci95_high": float(confidence.loc[term, 1]),
                            "converged": bool(fit.converged),
                            "warning": warning_text,
                            "error": "",
                        }
                    )
            except Exception as error:  # noqa: BLE001 - failed fits remain auditable
                rows.append(
                    {
                        **base,
                        "term": "",
                        "status": "fit_failed",
                        "coefficient": np.nan,
                        "std_error": np.nan,
                        "z_value": np.nan,
                        "p_value": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "converged": False,
                        "warning": "",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    output = pd.DataFrame(rows)
    if p_adjust is not None and not output.empty:
        output[f"p_{p_adjust}"] = adjust_pvalues(output["p_value"], method=p_adjust)
    return output


def pairwise_stager_correlations(
    hypnodensities: pd.DataFrame,
    stagers: Sequence[str] | None = None,
    group_by: Sequence[str] = (),
    alignment_keys: Sequence[str] = (
        "recording_id",
        "block_id",
        "epoch",
        "channel_set",
    ),
    method: str = "pearson",
    include_entropy: bool = True,
    p_adjust: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Correlate aligned epoch-level outputs for every pair of stagers."""

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as error:  # pragma: no cover - core feature dependency
        raise ImportError("Stager correlations require scipy.") from error
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    required = {"stager", *alignment_keys, *PROBABILITY_COLUMNS, *group_by}
    missing = sorted(required - set(hypnodensities.columns))
    if missing:
        raise ValueError(f"Hypnodensity table is missing columns: {missing}")

    table = add_hypnodensity_entropy(hypnodensities)
    selected = tuple(stagers or dict.fromkeys(table["stager"].astype(str)))
    unknown = sorted(set(selected) - set(table["stager"].astype(str)))
    if unknown:
        raise ValueError(f"Requested stagers are absent: {unknown}")
    duplicated = table.duplicated([*group_by, *alignment_keys, "stager"], keep=False)
    if duplicated.any():
        raise ValueError("Hypnodensity rows are not unique for the requested alignment keys.")

    grouped = [((), table)]
    if group_by:
        grouper = group_by[0] if len(group_by) == 1 else list(group_by)
        grouped = list(table.groupby(grouper, dropna=False, sort=False))
    metrics = list(PROBABILITY_COLUMNS)
    if include_entropy:
        metrics.append("entropy")
    rows = []
    for group_values, subset in grouped:
        if group_by and not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_metadata = dict(zip(group_by, group_values)) if group_by else {}
        indexed = {
            stager: subset.loc[subset["stager"].eq(stager)]
            .set_index(list(alignment_keys))
            .sort_index()
            for stager in selected
        }
        for first, second in combinations(selected, 2):
            joined = indexed[first][metrics].join(
                indexed[second][metrics],
                how="inner",
                lsuffix="_a",
                rsuffix="_b",
                validate="one_to_one",
            )
            for metric in metrics:
                x = joined[f"{metric}_a"].to_numpy(dtype=float)
                y = joined[f"{metric}_b"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 2 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
                    statistic = (
                        pearsonr(x[valid], y[valid])
                        if method == "pearson"
                        else spearmanr(x[valid], y[valid])
                    )
                    correlation, p_value = statistic.statistic, statistic.pvalue
                else:
                    correlation, p_value = np.nan, np.nan
                rows.append(
                    {
                        **group_metadata,
                        "stager_a": first,
                        "stager_b": second,
                        "metric": metric.removeprefix("prob_"),
                        "method": method,
                        "n": int(valid.sum()),
                        "correlation": correlation,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if p_adjust is not None and not result.empty:
        result[f"p_{p_adjust}"] = adjust_pvalues(result["p_value"], method=p_adjust)
    return result


def summarize_hypnodensities(
    hypnodensities: pd.DataFrame,
    group_by: Sequence[str] = (
        "subject",
        "condition",
        "experimental_label",
        "electrode",
        "stager",
    ),
) -> pd.DataFrame:
    """Return means, standard deviations and epoch counts for analysis groups."""

    table = add_hypnodensity_entropy(hypnodensities)
    missing = [column for column in group_by if column not in table]
    if missing:
        raise ValueError(f"Cannot summarize without grouping columns: {missing}")
    metrics = [*PROBABILITY_COLUMNS, "entropy"]
    grouped = table.groupby(list(group_by), dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    standard_deviation = grouped.std().add_suffix("_std")
    count = grouped.size().rename("n_epochs")
    return pd.concat([count, mean, standard_deviation], axis=1).reset_index()
