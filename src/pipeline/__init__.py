"""
ALPR Pipeline Module

Orchestrates the complete detection → recognition → tracking workflow.
"""

from .alpr_pipeline import ALPRPipeline
from .utils import (
    draw_tracks_on_frame,
    serialize_tracks,
    save_results_json,
    format_track_summary,
    create_side_by_side_comparison,
    log_pipeline_performance,
)

__all__ = [
    "ALPRPipeline",
    "draw_tracks_on_frame",
    "serialize_tracks",
    "save_results_json",
    "format_track_summary",
    "create_side_by_side_comparison",
    "log_pipeline_performance",
]
