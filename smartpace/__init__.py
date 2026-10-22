"""SmartPace: uncertainty-aware, explainable, edge-ready arrhythmia detection
with a simulated (non-clinical) decision-support layer."""
__version__ = "1.0.0"

from .adaptation import (
    adapt_linear_probe,
    evaluate_cross_dataset_rejection,
    evaluate_sampling_rate_control,
)
from .noise_policy import (
    AdaptiveReliabilityPolicy,
    BaseNoisePolicy,
    FixedDualGatePolicy,
    SQIOnlyPolicy,
    UncertaintyOnlyPolicy,
    ValCalibratedDualGatePolicy,
    evaluate_noise_type_breakdown,
    evaluate_policy_operating_point,
    generate_tradeoff_curve,
)
from .reliability import (
    AdaptiveAbstentionModule,
    EuclideanShiftDetector,
    MahalanobisShiftDetector,
    ReliabilityDecision,
    evaluate_four_configurations,
)
