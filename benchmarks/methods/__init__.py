"""
Baseline frame-selection methods registry.
"""

from .uniform import UniformSampling
from .random_sampling import RandomSampling
from .histogram import HistogramDedup
from .orb import ORBDedup
from .optical_flow import OpticalFlowPeaks
from .clip_dedup import CLIPOnlyDedup
from .kmeans import KMeansClustering
from .pipeline_variants import HIBPipelineBaseline, StaticPipelineBaseline

ALL_METHODS = {
    "uniform_1fps": UniformSampling,
    "random": RandomSampling,
    "histogram": HistogramDedup,
    "orb": ORBDedup,
    "optical_flow": OpticalFlowPeaks,
    "clip_only": CLIPOnlyDedup,
    "kmeans": KMeansClustering,
    "hib_pipeline": HIBPipelineBaseline,
    "static_pipeline": StaticPipelineBaseline,
}

__all__ = ["ALL_METHODS"]