"""DMT hypnodensity and EEG feature extraction."""

from .assembly import (
    AnalysisTables,
    assemble_outputs,
    feature_value_columns,
    join_epoch_features_hypnodensities,
)
from .batch import FileSelection, discover_recordings, run_batch, select_recordings
from .channels import ChannelSelection, load_electrode_mapping, select_analysis_electrodes
from .config import AnalysisConfig, ExperimentalWindow, analysis_config_to_mapping, load_config
from .epochs import ContinuousRun, EpochBlock, EpochMetadata, build_epoch_blocks
from .features import (
    FeatureResult,
    extract_block_features,
    extract_relative_bandpower,
    extract_yasa_staging_features,
)
from .io import (
    RecordingData,
    TrialData,
    load_fieldtrip_recording,
    read_fieldtrip_channels,
)
from .pipeline import RecordingResult, process_recording
from .plots import (
    plot_condition_change_violins,
    plot_electrode_variance_violins,
    plot_entropy_distribution,
    plot_feature_by_group,
    plot_hypnodensity_condition_violins,
    plot_hypnodensity_trajectories,
    plot_paired_condition_changes,
    plot_ranked_stage_features,
    plot_stage_feature_correlation_heatmap,
    plot_stage_feature_scatter,
    plot_stager_correlation_heatmap,
    save_figure,
)
from .runs import AnalysisRun, finalize_run, prepare_run
from .sleepfm import LocalSleepFMPredictor
from .staging import StagingResult, run_yasa, stage_block
from .stats import (
    add_hypnodensity_entropy,
    adjust_pvalues,
    fit_mixed_models,
    paired_condition_wilcoxon,
    pairwise_stager_correlations,
    prepare_epoch_cohen_d,
    prepare_treatment_effects,
    prepare_within_condition_changes,
    stage_feature_effect_correlations,
    summarize_hypnodensities,
)
from .storage import save_recording_result, save_table

__all__ = [
    "AnalysisConfig",
    "AnalysisRun",
    "AnalysisTables",
    "ChannelSelection",
    "ContinuousRun",
    "EpochBlock",
    "EpochMetadata",
    "ExperimentalWindow",
    "FeatureResult",
    "FileSelection",
    "LocalSleepFMPredictor",
    "RecordingData",
    "RecordingResult",
    "StagingResult",
    "TrialData",
    "add_hypnodensity_entropy",
    "adjust_pvalues",
    "analysis_config_to_mapping",
    "assemble_outputs",
    "build_epoch_blocks",
    "discover_recordings",
    "extract_block_features",
    "extract_relative_bandpower",
    "extract_yasa_staging_features",
    "feature_value_columns",
    "finalize_run",
    "fit_mixed_models",
    "join_epoch_features_hypnodensities",
    "load_config",
    "load_electrode_mapping",
    "load_fieldtrip_recording",
    "paired_condition_wilcoxon",
    "pairwise_stager_correlations",
    "plot_condition_change_violins",
    "plot_electrode_variance_violins",
    "plot_entropy_distribution",
    "plot_feature_by_group",
    "plot_hypnodensity_condition_violins",
    "plot_hypnodensity_trajectories",
    "plot_paired_condition_changes",
    "plot_ranked_stage_features",
    "plot_stage_feature_correlation_heatmap",
    "plot_stage_feature_scatter",
    "plot_stager_correlation_heatmap",
    "prepare_epoch_cohen_d",
    "prepare_run",
    "prepare_treatment_effects",
    "prepare_within_condition_changes",
    "process_recording",
    "read_fieldtrip_channels",
    "run_batch",
    "run_yasa",
    "save_figure",
    "save_recording_result",
    "save_table",
    "select_analysis_electrodes",
    "select_recordings",
    "stage_block",
    "stage_feature_effect_correlations",
    "summarize_hypnodensities",
]
