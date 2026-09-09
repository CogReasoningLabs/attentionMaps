"""Feature-oriented renderers for the dataset explorer tabs."""

from .comparison import render_comparison_tab
from .eda import render_eda_tab, render_notes_tab
from .evaluation import render_evaluation_tab, render_nlue_tab
from .language import render_tokenizer_tab, render_wordcloud_tab
from .local_models import render_local_inference_tab
from .manifest import render_manifest_tab
from .overview import render_details_tab, render_sample_tab

__all__ = [
    "render_comparison_tab",
    "render_details_tab",
    "render_eda_tab",
    "render_evaluation_tab",
    "render_local_inference_tab",
    "render_manifest_tab",
    "render_nlue_tab",
    "render_notes_tab",
    "render_sample_tab",
    "render_tokenizer_tab",
    "render_wordcloud_tab",
]
