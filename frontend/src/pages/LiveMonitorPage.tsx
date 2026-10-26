import React from 'react';
import { Layers, ShieldCheck, RefreshCw, AlertCircle } from 'lucide-react';
import { ECGWaveformCanvas } from '../components/ECGWaveformCanvas';
import { LiveMetrics } from '../components/LiveMetrics';
import { RhythmPredictionCard } from '../components/RhythmPredictionCard';
import { ReliabilityTriGate } from '../components/ReliabilityTriGate';
import { PlaybackControls } from '../components/PlaybackControls';
import { EventTimeline } from '../components/EventTimeline';
import { EdgeBenchmarkPanel } from '../components/EdgeBenchmarkPanel';
import {
  RecordItem,
  StreamStatus,
  EventItem,
  PredictionData,
  DecisionData,
  VitalsData,
  InferenceModeInfo,
} from '../types';

interface LiveMonitorPageProps {
  samples: { value: number; is_rpeak: boolean; timestamp: number }[];
  status: StreamStatus;
  records: RecordItem[];
  selectedRecord: string;
  onSelectRecord: (name: string) => void;
  onPlay: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onSpeedChange: (speed: number) => void;
  mode: 'replay' | 'sensor_adapter';
  onModeChange: (mode: 'replay' | 'sensor_adapter') => void;
  latestPrediction?: PredictionData;
  latestDecision?: DecisionData;
  latestVitals?: VitalsData;
  latestShiftScore?: number;
  latencyMs?: number;
  events: EventItem[];
  onSelectEventForExplanation: (event: EventItem) => void;
  onRefreshEvents: () => void;
  inferenceMode: 'single' | 'ensemble';
  inferenceModeInfo: InferenceModeInfo | null;
  onInferenceModeToggle: (mode: 'single' | 'ensemble') => void;
  isSwitchingMode: boolean;
  modeSwitchError: string | null;
  latestEnsembleInfo?: any;
}

export const LiveMonitorPage: React.FC<LiveMonitorPageProps> = ({
  samples,
  status,
  records,
  selectedRecord,
  onSelectRecord,
  onPlay,
  onPause,
  onResume,
  onStop,
  onSpeedChange,
  mode,
  onModeChange,
  latestPrediction,
  latestDecision,
  latestVitals,
  latestShiftScore,
  latencyMs,
  events,
  onSelectEventForExplanation,
  onRefreshEvents,
  inferenceMode,
  inferenceModeInfo,
  onInferenceModeToggle,
  isSwitchingMode,
  modeSwitchError,
  latestEnsembleInfo,
}) => {
  return (
    <div className="space-y-4">
      {/* 1. Playback & Replay Controls */}
      <PlaybackControls
        status={status}
        records={records}
        selectedRecord={selectedRecord}
        onSelectRecord={onSelectRecord}
        onPlay={onPlay}
        onPause={onPause}
        onResume={onResume}
        onStop={onStop}
        onSpeedChange={onSpeedChange}
        mode={mode}
        onModeChange={onModeChange}
      />

      {/* 2. Inference Mode HUD Bar (Single Seed vs Deep Ensemble) */}
      <div className="bg-[#0f1422] border border-slate-800 rounded-xl p-3.5 shadow-md flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs font-mono">
            <button
              onClick={() => onInferenceModeToggle('single')}
              disabled={isSwitchingMode || inferenceMode === 'single'}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                inferenceMode === 'single'
                  ? 'bg-blue-600/30 text-blue-300 border border-blue-500/50 font-bold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 disabled:opacity-50'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Single Seed (Low Latency)</span>
            </button>

            <button
              onClick={() => onInferenceModeToggle('ensemble')}
              disabled={isSwitchingMode || inferenceMode === 'ensemble' || !inferenceModeInfo?.ensemble_available}
              title={
                !inferenceModeInfo?.ensemble_available
                  ? inferenceModeInfo?.ensemble_unavailable_reason || 'Ensemble checkpoints missing'
                  : '3-Member Deep Ensemble'
              }
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                inferenceMode === 'ensemble'
                  ? 'bg-purple-600/30 text-purple-300 border border-purple-500/50 font-bold shadow-sm'
                  : inferenceModeInfo?.ensemble_available
                  ? 'text-slate-400 hover:text-slate-200 disabled:opacity-50'
                  : 'text-slate-600 cursor-not-allowed'
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Deep Ensemble (3 Models)</span>
            </button>
          </div>

          {isSwitchingMode && (
            <div className="flex items-center gap-1.5 text-xs text-indigo-400 animate-pulse font-mono">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Switching Mode...</span>
            </div>
          )}

          {modeSwitchError && (
            <span className="text-xs text-rose-400 font-mono flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5" />
              {modeSwitchError}
            </span>
          )}
        </div>

        {/* Active Model & Ensemble Status Display */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <span className="text-slate-400">Architecture:</span>
          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
            ResNet-1D
          </span>

          <span
            className={`px-2 py-0.5 rounded border font-semibold flex items-center gap-1.5 ${
              inferenceMode === 'ensemble'
                ? 'bg-purple-500/10 text-purple-300 border-purple-500/30'
                : 'bg-blue-500/10 text-blue-300 border-blue-500/30'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                inferenceMode === 'ensemble' ? 'bg-purple-400 animate-pulse' : 'bg-blue-400'
              }`}
            ></span>
            {inferenceMode === 'ensemble' ? '3 Loaded Members (s0, s1, s2)' : '1 Model (s0)'}
          </span>

          {inferenceMode === 'ensemble' && latestEnsembleInfo && (
            <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
              Disagreement: {(latestEnsembleInfo.disagreement * 100).toFixed(0)}% (Var:{' '}
              {latestEnsembleInfo.probability_variance?.toFixed(4) || '0.0000'})
            </span>
          )}
        </div>
      </div>

      {/* 3. Real-Time Waveform Monitor (Incremental Causal Peak Markers) */}
      <ECGWaveformCanvas
        samples={samples}
        currentRecord={selectedRecord}
        leadName="MLII"
        samplingRate={250}
      />

      {/* 4. Live Metrics Row */}
      <LiveMetrics vitals={latestVitals} latencyMs={latencyMs} />

      {/* 5. Dual Center Grid: Rhythm Classification & Tri-Gate Reliability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RhythmPredictionCard prediction={latestPrediction} />

        <ReliabilityTriGate
          decision={latestDecision}
          sqiScore={latestVitals?.sqi_scores?.composite_sqi}
          uncertaintyScore={latestPrediction?.uncertainty_score}
          shiftScore={latestShiftScore}
        />
      </div>

      {/* 6. Live FP32 vs ONNX INT8 Benchmark Panel */}
      <EdgeBenchmarkPanel />

      {/* 7. Real-Time Event Log */}
      <EventTimeline
        events={events}
        onSelectEventForExplanation={onSelectEventForExplanation}
        onRefresh={onRefreshEvents}
      />
    </div>
  );
};
