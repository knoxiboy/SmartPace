import React from 'react';
import { ECGWaveformCanvas } from '../components/ECGWaveformCanvas';
import { LiveMetrics } from '../components/LiveMetrics';
import { RhythmPredictionCard } from '../components/RhythmPredictionCard';
import { ReliabilityTriGate } from '../components/ReliabilityTriGate';
import { PlaybackControls } from '../components/PlaybackControls';
import { EventTimeline } from '../components/EventTimeline';
import {
  RecordItem,
  StreamStatus,
  EventItem,
  PredictionData,
  DecisionData,
  VitalsData
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
}) => {
  return (
    <div className="space-y-4">
      {/* 1. Playback & Mode Controls */}
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

      {/* 2. Real-Time Waveform Monitor */}
      <ECGWaveformCanvas
        samples={samples}
        currentRecord={selectedRecord}
        leadName="MLII"
        samplingRate={250}
      />

      {/* 3. Live Metrics Row */}
      <LiveMetrics vitals={latestVitals} latencyMs={latencyMs} />

      {/* 4. Dual Center Grid: Rhythm Classification & Tri-Gate Reliability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RhythmPredictionCard prediction={latestPrediction} />

        <ReliabilityTriGate
          decision={latestDecision}
          sqiScore={latestVitals?.sqi_scores?.composite_sqi}
          uncertaintyScore={latestPrediction?.uncertainty_score}
          shiftScore={latestShiftScore}
        />
      </div>

      {/* 5. Real-Time Event Log */}
      <EventTimeline
        events={events}
        onSelectEventForExplanation={onSelectEventForExplanation}
        onRefresh={onRefreshEvents}
      />
    </div>
  );
};
