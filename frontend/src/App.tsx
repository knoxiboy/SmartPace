import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Navbar, ActiveTab } from './components/Navbar';
import { LiveMonitorPage } from './pages/LiveMonitorPage';
import { ExplainabilityLabPage } from './pages/ExplainabilityLabPage';
import { StressTestPage } from './pages/StressTestPage';
import { ResearchEvaluationPage } from './pages/ResearchEvaluationPage';
import { SystemInfoPage } from './pages/SystemInfoPage';
import { api } from './services/api';
import {
  RecordItem,
  StreamStatus,
  EventItem,
  PredictionData,
  DecisionData,
  VitalsData,
  ArrhythmiaClass,
  DecisionStatus,
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('monitor');
  const [isWsConnected, setIsWsConnected] = useState<boolean>(false);
  const [records, setRecords] = useState<RecordItem[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<string>('100');
  const [mode, setMode] = useState<'replay' | 'sensor_adapter'>('replay');

  // Streaming State
  const [streamStatus, setStreamStatus] = useState<StreamStatus>({
    is_streaming: false,
    record_name: '100',
    playback_speed: 1.0,
    current_sample: 0,
    total_samples: 450000,
    elapsed_sec: 0,
    total_duration_sec: 1800,
    mode: 'replay',
    window_count: 0,
    active_clients: 0,
  });

  // Real-time rolling sample buffer for canvas (kept up to 3000 samples)
  const [samples, setSamples] = useState<{ value: number; is_rpeak: boolean; timestamp: number }[]>([]);

  // Inference state
  const [latestPrediction, setLatestPrediction] = useState<PredictionData | undefined>(undefined);
  const [latestDecision, setLatestDecision] = useState<DecisionData | undefined>(undefined);
  const [latestVitals, setLatestVitals] = useState<VitalsData | undefined>(undefined);
  const [latestShiftScore, setLatestShiftScore] = useState<number | undefined>(undefined);
  const [latencyMs, setLatencyMs] = useState<number>(12.4);

  // Events history
  const [events, setEvents] = useState<EventItem[]>([]);
  const [selectedEventForExplanation, setSelectedEventForExplanation] = useState<EventItem | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // Load records and initial events on mount
  useEffect(() => {
    api.getRecords()
      .then((recs) => {
        const arr = Array.isArray(recs) ? recs : [];
        setRecords(arr);
        if (arr.length > 0) {
          setSelectedRecord(arr[0].name);
        }
      })
      .catch((err) => {
        console.error('Failed to load records:', err);
        setRecords([]);
      });

    loadEvents();
  }, []);

  const loadEvents = useCallback(() => {
    api.getEvents({ limit: 50 })
      .then((res) => {
        setEvents(res.events || []);
      })
      .catch(console.error);
  }, []);

  // WebSocket Telemetry Connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/ecg`;

    let ws: WebSocket;
    let reconnectTimer: any = null;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          // 1. Continuous ECG waveform telemetry chunk
          if (msg.type === 'ecg_chunk' || msg.type === 'sample') {
            const rawSamples: number[] = Array.isArray(msg.samples)
              ? msg.samples
              : typeof msg.value === 'number'
              ? [msg.value]
              : [];

            if (rawSamples.length > 0) {
              const now = msg.timestamp || Date.now();
              const newPoints = rawSamples.map((v) => ({
                value: Number(v),
                is_rpeak: false,
                timestamp: now,
              }));

              setSamples((prev) => {
                const merged = [...prev, ...newPoints];
                return merged.length > 3000 ? merged.slice(merged.length - 3000) : merged;
              });
            }
          }

          // 2. Window ML inference decision & Tri-Gate reliability result
          else if (msg.type === 'inference_result' || msg.type === 'window_inference') {
            const data = msg.data || msg;
            const pred = data.prediction || {};
            const rel = data.reliability || {};
            const vit = data.vitals || {};
            const dec = data.decision || {};

            const predClass = (pred.predicted_class || data.predicted_class || 'NSR') as ArrhythmiaClass;
            const conf = pred.confidence ?? data.confidence ?? 0.95;
            const probs = pred.probabilities || data.class_probabilities || { [predClass]: conf };
            const entropy = pred.entropy ?? data.predictive_entropy ?? 0.08;
            const uncert = pred.uncertainty_score ?? data.predictive_entropy ?? 0.08;

            const predData: PredictionData = {
              predicted_class: predClass,
              confidence: conf,
              probabilities: probs,
              entropy: entropy,
              uncertainty_score: uncert,
            };

            const decStatus = (dec.status || rel.action || (rel.is_accepted ? 'ACCEPT' : 'REJECT_QUALITY')) as DecisionStatus;
            const decReason = dec.reason || rel.reason || 'Tri-Gate assessment completed';
            const decAccepted = dec.accepted ?? rel.is_accepted ?? true;
            const decFailures = dec.gate_failures || (rel.is_accepted ? [] : [rel.reason || 'Gate condition failed']);

            const decisionData: DecisionData = {
              status: decStatus,
              reason: decReason,
              accepted: decAccepted,
              gate_failures: decFailures,
            };

            const vitalsData: VitalsData = {
              heart_rate_bpm: Math.round(vit.heart_rate_bpm ?? 72),
              rr_intervals_ms: vit.rr_intervals_ms || [],
              sqi_scores: {
                kurtosis_sqi: vit.sqi_scores?.kurtosis_sqi ?? rel.sqi_kurtosis ?? 4.5,
                skewness_sqi: vit.sqi_scores?.skewness_sqi ?? 0.1,
                power_sqi: vit.sqi_scores?.power_sqi ?? 0.9,
                relative_power_sqi: vit.sqi_scores?.relative_power_sqi ?? 0.85,
                baseline_wander_sqi: vit.sqi_scores?.baseline_wander_sqi ?? (1.0 - (rel.sqi_bw_ratio || 0.05)),
                high_freq_sqi: vit.sqi_scores?.high_freq_sqi ?? (1.0 - (rel.sqi_hf_ratio || 0.05)),
                composite_sqi: vit.sqi_scores?.composite_sqi ?? rel.sqi_score ?? 0.92,
              },
            };

            const shift = data.shift_score ?? rel.mahalanobis_distance ?? 12.0;
            const lat = data.latency_ms ?? data.latency?.total_processing_ms ?? 12.4;

            setLatestPrediction(predData);
            setLatestDecision(decisionData);
            setLatestVitals(vitalsData);
            setLatestShiftScore(shift);
            setLatencyMs(lat);

            // Mark detected R-peaks in recent rolling canvas buffer
            if (Array.isArray(vit.r_peak_indices) && vit.r_peak_indices.length > 0) {
              setSamples((prev) => {
                if (prev.length === 0) return prev;
                const windowStartInPrev = Math.max(0, prev.length - 2500);
                const rSet = new Set(vit.r_peak_indices.map((idx: number) => windowStartInPrev + idx));
                return prev.map((s, idx) => ({
                  ...s,
                  is_rpeak: s.is_rpeak || rSet.has(idx),
                }));
              });
            }

            // Prepend new event to timeline
            const newEvt: EventItem = {
              id: data.event_id || `ev_${Date.now()}`,
              timestamp: new Date().toISOString(),
              record_name: data.record_name || data.record_id || streamStatus.record_name || 'MIT-BIH',
              window_start_sec: data.window_start_sec ?? streamStatus.elapsed_sec ?? 0,
              window_end_sec: data.window_end_sec ?? ((streamStatus.elapsed_sec ?? 0) + 10),
              predicted_class: predClass,
              confidence: conf,
              status: decStatus,
              composite_sqi: vitalsData.sqi_scores.composite_sqi,
              uncertainty_score: uncert,
              shift_score: shift,
              heart_rate_bpm: vitalsData.heart_rate_bpm,
              accepted: decAccepted,
              gate_failures: decFailures,
              reason: decReason,
            };

            setEvents((prev) => [newEvt, ...prev.slice(0, 99)]);
          }

          // 3. Status updates and streaming engine lifecycle events
          else if (
            msg.type === 'status' ||
            msg.type === 'stream_status' ||
            msg.type === 'connection_established'
          ) {
            const st = msg.status || msg.data;
            if (st) {
              setStreamStatus((prev) => ({
                ...prev,
                ...st,
                elapsed_sec: st.elapsed_sec ?? st.current_time_s ?? prev.elapsed_sec,
                total_duration_sec: st.total_duration_sec ?? st.duration_s ?? prev.total_duration_sec,
                playback_speed: st.playback_speed ?? st.speed ?? prev.playback_speed,
                is_streaming: st.is_streaming ?? st.is_playing ?? prev.is_streaming,
              }));
            }
          } else if (msg.type === 'stream_started' || msg.type === 'stream_resumed') {
            setStreamStatus((prev) => ({
              ...prev,
              is_streaming: true,
              is_paused: false,
              record_name: msg.record_name || prev.record_name,
              playback_speed: msg.speed ?? prev.playback_speed,
              elapsed_sec: msg.start_time_s ?? msg.current_time_s ?? prev.elapsed_sec,
            }));
          } else if (msg.type === 'stream_paused') {
            setStreamStatus((prev) => ({
              ...prev,
              is_streaming: true,
              is_paused: true,
              elapsed_sec: msg.current_time_s ?? prev.elapsed_sec,
            }));
          } else if (msg.type === 'stream_stopped') {
            setStreamStatus((prev) => ({
              ...prev,
              is_streaming: false,
              is_paused: false,
              elapsed_sec: 0,
            }));
          } else if (msg.type === 'stream_speed_changed') {
            setStreamStatus((prev) => ({
              ...prev,
              playback_speed: msg.speed ?? prev.playback_speed,
            }));
          }
        } catch (err) {
          console.error('WS parse error:', err);
        }
      };

      ws.onclose = () => {
        setIsWsConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        ws.close();
      };
    };

    connect();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Poll stream status every 3s as backup
  useEffect(() => {
    const interval = setInterval(() => {
      api.getStreamStatus()
        .then((s) => setStreamStatus(s))
        .catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Playback handlers
  const handlePlay = async () => {
    try {
      await api.startStream({
        record_name: selectedRecord,
        playback_speed: streamStatus.playback_speed,
      });
      const s = await api.getStreamStatus();
      setStreamStatus(s);
    } catch (err) {
      console.error('Start stream error:', err);
    }
  };

  const handlePause = async () => {
    try {
      await api.pauseStream();
      setStreamStatus((prev) => ({ ...prev, is_streaming: false }));
    } catch (err) {
      console.error('Pause stream error:', err);
    }
  };

  const handleResume = async () => {
    try {
      await api.resumeStream();
      setStreamStatus((prev) => ({ ...prev, is_streaming: true }));
    } catch (err) {
      console.error('Resume stream error:', err);
    }
  };

  const handleStop = async () => {
    try {
      await api.stopStream();
      const s = await api.getStreamStatus();
      setStreamStatus(s);
      setSamples([]);
    } catch (err) {
      console.error('Stop stream error:', err);
    }
  };

  const handleSpeedChange = async (speed: number) => {
    try {
      await api.setSpeed(speed);
      setStreamStatus((prev) => ({ ...prev, playback_speed: speed }));
    } catch (err) {
      console.error('Set speed error:', err);
    }
  };

  const handleSelectEventForExplanation = (evt: EventItem) => {
    setSelectedEventForExplanation(evt);
    setActiveTab('explainability');
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isWsConnected={isWsConnected}
        isStreaming={streamStatus.is_streaming}
        recordName={selectedRecord}
      />

      {/* Main Content View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'monitor' && (
          <LiveMonitorPage
            samples={samples}
            status={streamStatus}
            records={records}
            selectedRecord={selectedRecord}
            onSelectRecord={(name) => {
              setSelectedRecord(name);
              setStreamStatus((prev) => ({ ...prev, record_name: name }));
            }}
            onPlay={handlePlay}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            onSpeedChange={handleSpeedChange}
            mode={mode}
            onModeChange={setMode}
            latestPrediction={latestPrediction}
            latestDecision={latestDecision}
            latestVitals={latestVitals}
            latestShiftScore={latestShiftScore}
            latencyMs={latencyMs}
            events={events}
            onSelectEventForExplanation={handleSelectEventForExplanation}
            onRefreshEvents={loadEvents}
          />
        )}

        {activeTab === 'explainability' && (
          <ExplainabilityLabPage
            selectedEventForExplanation={selectedEventForExplanation}
          />
        )}

        {activeTab === 'stress' && <StressTestPage records={records} />}

        {activeTab === 'evaluation' && <ResearchEvaluationPage />}

        {activeTab === 'system' && <SystemInfoPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0c101a] py-4 text-center text-xs font-mono text-slate-500">
        <p>SmartPace Live • Biomedical Signal Processing & Machine Learning Research Platform</p>
        <p className="text-[10px] text-slate-600 mt-1">
          PhysioNet MIT-BIH &amp; NSTDB • ResNet-1D AAMI Benchmark • MIT License
        </p>
      </footer>
    </div>
  );
};

export default App;
