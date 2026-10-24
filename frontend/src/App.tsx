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
  VitalsData
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
        setRecords(recs);
        if (recs.length > 0) {
          setSelectedRecord(recs[0].name);
        }
      })
      .catch(console.error);

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
          if (msg.type === 'sample') {
            setSamples((prev) => {
              const next = [
                ...prev,
                {
                  value: msg.value,
                  is_rpeak: msg.is_rpeak,
                  timestamp: msg.timestamp,
                },
              ];
              if (next.length > 3000) {
                return next.slice(next.length - 3000);
              }
              return next;
            });
          } else if (msg.type === 'window_inference') {
            const data = msg.data;
            setLatestPrediction(data.prediction);
            setLatestDecision(data.decision);
            setLatestVitals(data.vitals);
            setLatestShiftScore(data.shift_score);
            if (data.latency_ms) setLatencyMs(data.latency_ms);

            // Prepend new event to timeline
            const newEvt: EventItem = {
              id: data.event_id,
              timestamp: new Date().toISOString(),
              record_name: data.record_name,
              window_start_sec: data.window_start_sec,
              window_end_sec: data.window_end_sec,
              predicted_class: data.prediction.predicted_class,
              confidence: data.prediction.confidence,
              status: data.decision.status,
              composite_sqi: data.vitals.sqi_scores.composite_sqi,
              uncertainty_score: data.prediction.uncertainty_score,
              shift_score: data.shift_score,
              heart_rate_bpm: data.vitals.heart_rate_bpm,
              accepted: data.decision.accepted,
              gate_failures: data.decision.gate_failures,
              reason: data.decision.reason,
            };

            setEvents((prev) => [newEvt, ...prev.slice(0, 99)]);
          } else if (msg.type === 'status') {
            setStreamStatus(msg.data);
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
