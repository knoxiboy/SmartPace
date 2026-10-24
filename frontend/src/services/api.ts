import {
  RecordItem,
  StreamStatus,
  EventItem,
  ExplainResponse,
  StressTestResponse,
  FailureCase,
  PolicyResult,
  TradeoffCurve,
  SystemInfo
} from '../types';

const API_BASE = '/api';

export const api = {
  // Records & Stream Controls
  async getRecords(): Promise<RecordItem[]> {
    const res = await fetch(`${API_BASE}/records`);
    if (!res.ok) throw new Error('Failed to fetch records');
    return res.json();
  },

  async startStream(params: {
    record_name?: string;
    playback_speed?: number;
    start_sample?: number;
    lead_idx?: number;
  }): Promise<{ status: string; detail: string }> {
    const res = await fetch(`${API_BASE}/stream/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to start stream');
    return res.json();
  },

  async pauseStream(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/stream/pause`, { method: 'POST' });
    return res.json();
  },

  async resumeStream(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/stream/resume`, { method: 'POST' });
    return res.json();
  },

  async stopStream(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/stream/stop`, { method: 'POST' });
    return res.json();
  },

  async setSpeed(speed: number): Promise<{ status: string; speed: number }> {
    const res = await fetch(`${API_BASE}/stream/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed }),
    });
    return res.json();
  },

  async getStreamStatus(): Promise<StreamStatus> {
    const res = await fetch(`${API_BASE}/stream/status`);
    if (!res.ok) throw new Error('Failed to fetch stream status');
    return res.json();
  },

  // Events & Inference
  async getEvents(params?: {
    limit?: number;
    status?: string;
    class_label?: string;
  }): Promise<{ total: number; events: EventItem[] }> {
    const query = new URLSearchParams();
    if (params?.limit) query.set('limit', params.limit.toString());
    if (params?.status) query.set('status', params.status);
    if (params?.class_label) query.set('class_label', params.class_label);

    const res = await fetch(`${API_BASE}/events?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch events');
    return res.json();
  },

  async getEventById(id: string): Promise<any> {
    const res = await fetch(`${API_BASE}/events/${id}`);
    if (!res.ok) throw new Error('Failed to fetch event detail');
    return res.json();
  },

  async getEventStatistics(): Promise<any> {
    const res = await fetch(`${API_BASE}/events/statistics`);
    if (!res.ok) throw new Error('Failed to fetch event statistics');
    return res.json();
  },

  // Explainability
  async getExplanation(params: {
    event_id?: string;
    signal?: number[];
    target_class?: string;
    n_steps?: number;
  }): Promise<ExplainResponse> {
    const res = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to compute explanation');
    return res.json();
  },

  async runStressTest(params: {
    event_id?: string;
    signal?: number[];
    noise_kind: string;
    snr_db: number;
  }): Promise<StressTestResponse> {
    const res = await fetch(`${API_BASE}/explain/stress_test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to run explanation stress test');
    return res.json();
  },

  async getFailureCases(): Promise<{ count: number; cases: FailureCase[] }> {
    const res = await fetch(`${API_BASE}/explain/failure_cases`);
    if (!res.ok) throw new Error('Failed to fetch failure cases');
    return res.json();
  },

  // Stress & Reliability
  async getNoiseTypes(): Promise<{ noise_types: string[]; default_snr_db: number }> {
    const res = await fetch(`${API_BASE}/stress/noise_types`);
    if (!res.ok) throw new Error('Failed to fetch noise types');
    return res.json();
  },

  async corruptSignal(params: {
    signal?: number[];
    record_name?: string;
    noise_type: string;
    snr_db: number;
  }): Promise<{
    clean_signal: number[];
    noisy_signal: number[];
    noise_type: string;
    snr_db: number;
    clean_sqi: any;
    noisy_sqi: any;
  }> {
    const res = await fetch(`${API_BASE}/stress/corrupt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to corrupt signal');
    return res.json();
  },

  async comparePolicies(params: {
    signal?: number[];
    record_name?: string;
    noise_type: string;
    snr_db: number;
  }): Promise<{
    snr_db: number;
    noise_type: string;
    clean_prediction: any;
    noisy_prediction: any;
    noisy_shift_score: number;
    noisy_sqi: any;
    policies: Record<string, PolicyResult>;
  }> {
    const res = await fetch(`${API_BASE}/stress/compare_policies`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to compare policies');
    return res.json();
  },

  async getTradeoffCurves(): Promise<{ curves: TradeoffCurve[] }> {
    const res = await fetch(`${API_BASE}/stress/tradeoff_curves`);
    if (!res.ok) throw new Error('Failed to fetch risk-coverage curves');
    return res.json();
  },

  // Research Evaluation
  async getExperiments(): Promise<any> {
    const res = await fetch(`${API_BASE}/evaluation/experiments`);
    if (!res.ok) throw new Error('Failed to fetch experiment list');
    return res.json();
  },

  async getExperimentResults(expName: string): Promise<any> {
    const res = await fetch(`${API_BASE}/evaluation/results/${expName}`);
    if (!res.ok) throw new Error(`Failed to fetch results for ${expName}`);
    return res.json();
  },

  // System Info
  async getSystemInfo(): Promise<SystemInfo> {
    const res = await fetch(`${API_BASE}/system/info`);
    if (!res.ok) throw new Error('Failed to fetch system info');
    return res.json();
  },
};
