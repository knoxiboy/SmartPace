export type ArrhythmiaClass =
  | 'N'
  | 'S'
  | 'V'
  | 'F'
  | 'Q'
  | 'NSR'
  | 'SBRAD'
  | 'STACH'
  | 'AF'
  | 'PVC';

export const CLASS_LABELS: Record<string, string> = {
  N: 'Normal Sinus Rhythm',
  S: 'Supraventricular Ectopic Beat (SVEB)',
  V: 'Ventricular Ectopic Beat (VEB)',
  F: 'Fusion Beat',
  Q: 'Unclassifiable / Paced Beat',
  NSR: 'Normal Sinus Rhythm (NSR)',
  SBRAD: 'Sinus Bradycardia (<60 bpm)',
  STACH: 'Sinus Tachycardia (>100 bpm)',
  AF: 'Atrial Fibrillation (AF)',
  PVC: 'Premature Ventricular Contraction',
};

export const CLASS_COLORS: Record<string, string> = {
  N: '#10b981', // green
  S: '#38bdf8', // light blue
  V: '#f43f5e', // red
  F: '#f59e0b', // amber
  Q: '#a855f7', // purple
  NSR: '#10b981', // emerald
  SBRAD: '#38bdf8', // light blue
  STACH: '#f59e0b', // amber
  AF: '#a855f7', // purple
  PVC: '#f43f5e', // rose
};

export type DecisionStatus = 'ACCEPT' | 'REJECT_QUALITY' | 'ABSTAIN_UNCERTAIN' | 'ABSTAIN_UNFAMILIAR';

export interface RecordItem {
  name: string;
  source: string;
  duration_sec: number;
  sample_rate: number;
  leads: string[];
}

export interface StreamStatus {
  is_streaming: boolean;
  record_name: string;
  playback_speed: number;
  current_sample: number;
  total_samples: number;
  elapsed_sec: number;
  total_duration_sec: number;
  mode: 'replay' | 'sensor_adapter';
  window_count: number;
  active_clients: number;
}

export interface VitalsData {
  heart_rate_bpm: number;
  rr_intervals_ms: number[];
  sqi_scores: {
    kurtosis_sqi: number;
    skewness_sqi: number;
    power_sqi: number;
    relative_power_sqi: number;
    baseline_wander_sqi: number;
    high_freq_sqi: number;
    composite_sqi: number;
  };
}

export interface PredictionData {
  predicted_class: ArrhythmiaClass;
  confidence: number;
  probabilities: Record<ArrhythmiaClass, number>;
  entropy: number;
  uncertainty_score: number;
}

export interface DecisionData {
  status: DecisionStatus;
  reason: string;
  accepted: boolean;
  gate_failures: string[];
}

export interface WindowInferencePayload {
  event_id: string;
  record_name: string;
  window_start_sec: number;
  window_end_sec: number;
  prediction: PredictionData;
  decision: DecisionData;
  vitals: VitalsData;
  shift_score: number;
  latency_ms: number;
  r_peaks: number[];
  lead_name: string;
}

export interface EventItem {
  id: string;
  timestamp: string;
  record_name: string;
  window_start_sec: number;
  window_end_sec: number;
  predicted_class: ArrhythmiaClass;
  confidence: number;
  status: DecisionStatus;
  composite_sqi: number;
  uncertainty_score: number;
  shift_score: number;
  heart_rate_bpm: number;
  accepted: boolean;
  gate_failures: string[];
  reason: string;
}

export interface AnatomicalRegion {
  name: string;
  indices: [number, number];
  relative_importance: number;
  clinical_significance: string;
}

export interface ExplainResponse {
  event_id?: string;
  predicted_class: ArrhythmiaClass;
  confidence: number;
  decision: DecisionData;
  clean_signal: number[];
  attributions: number[];
  anatomical_regions: AnatomicalRegion[];
  method: string;
  baseline_mode: string;
  convergence_delta?: number;
}

export interface StressTestResponse {
  event_id: string;
  baseline_prediction: PredictionData;
  baseline_decision: DecisionData;
  noisy_prediction: PredictionData;
  noisy_decision: DecisionData;
  attribution_stability_rho: number;
  counterfactual_flips: boolean;
  anatomical_shift: {
    baseline_dominant_region: string;
    noisy_dominant_region: string;
    qrs_focus_drop_pct: number;
  };
}

export interface FailureCase {
  case_id: string;
  type: string;
  title: string;
  description: string;
  record_name: string;
  window_sec: number;
  prediction: PredictionData;
  decision: DecisionData;
  shift_score: number;
  sqi_scores: Record<string, number>;
  clinical_takeaway: string;
  signal_preview?: number[];
}

export interface PolicyResult {
  policy_name: string;
  decision: DecisionData;
  is_accepted: boolean;
  abstention_type: string;
  confidence: number;
  reliability_score: number;
}

export interface TradeoffCurve {
  policy: string;
  points: { coverage: number; empirical_risk: number; threshold: number }[];
  aurc: number;
}

export interface SystemInfo {
  app_name: string;
  version: string;
  device: string;
  active_model: {
    name: string;
    type: string;
    checkpoint: string;
    parameters: number;
    input_shape: number[];
  };
  calibrated_thresholds: {
    sqi_composite_threshold: number;
    uncertainty_tau: number;
    mahalanobis_lambda: number;
    temperature_t: number;
  };
  mitdb_records_available: number;
  noise_nstdb_available: boolean;
  disclaimer: string;
}

export interface InferenceModeInfo {
  mode: 'single' | 'ensemble';
  active_models: number;
  ensemble_available: boolean;
  ensemble_unavailable_reason?: string | null;
  primary_model?: string;
  ensemble_members?: string[];
}

export interface BenchmarkResult {
  status: string;
  benchmark_type: string;
  test_windows_evaluated: number;
  repetitions_per_window: number;
  total_timed_inferences: number;
  warmup_runs: number;
  timing_unit: string;
  pytorch_fp32: {
    model_file: string;
    size_kb: number;
    device: string;
    mean_latency_ms: number;
    median_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
  };
  onnx_int8: {
    model_file: string;
    size_kb: number;
    execution_provider: string;
    all_providers: string[];
    mean_latency_ms: number;
    median_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
  };
  comparison: {
    speedup_ratio_median: number;
    speedup_ratio_mean: number;
    compression_ratio: number;
    prediction_agreement_rate: number;
    prediction_agreement_pct: number;
    matching_windows: number;
    mean_probability_drift: number;
    max_probability_drift: number;
  };
  scientific_disclaimer: string;
}

export interface PeakEvent {
  sample_index: number;
  timestamp: number;
  detection_sample_idx: number;
  detection_time_s: number;
  amplitude: number;
  delay_samples: number;
  delay_ms: number;
  confidence: number;
}

