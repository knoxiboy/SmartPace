import React from 'react';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  FastForward,
  Database,
  Radio,
  Sliders
} from 'lucide-react';
import { RecordItem, StreamStatus } from '../types';

interface PlaybackControlsProps {
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
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
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
}) => {
  const speeds = [0.25, 0.5, 1.0, 2.0, 5.0];

  const formatTime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  const progressPct =
    status.total_duration_sec > 0
      ? (status.elapsed_sec / status.total_duration_sec) * 100
      : 0;

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Mode & Record Selection */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Mode Switcher */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
            <button
              onClick={() => onModeChange('replay')}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded-md transition-all ${
                mode === 'replay'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>Mode A: Replay</span>
            </button>

            <button
              onClick={() => onModeChange('sensor_adapter')}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded-md transition-all ${
                mode === 'sensor_adapter'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Radio className="w-3.5 h-3.5" />
              <span>Mode B: Sensor Adapter</span>
            </button>
          </div>

          {/* Record Selector Dropdown */}
          <div className="flex items-center space-x-2">
            <label className="text-xs text-slate-400 font-mono">Record:</label>
            <select
              value={selectedRecord}
              onChange={(e) => onSelectRecord(e.target.value)}
              disabled={status.is_streaming}
              className="bg-slate-900 border border-slate-700 text-cyan-300 text-xs font-mono rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-emerald-500 disabled:opacity-50"
            >
              {records.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name} ({r.source} • {r.leads.join('/')})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Center: Playback Controls (Play/Pause/Stop) */}
        <div className="flex items-center space-x-2 self-center">
          {!status.is_streaming ? (
            <button
              onClick={onPlay}
              className="flex items-center space-x-1.5 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-md shadow-emerald-600/30 transition-all"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>START STREAM</span>
            </button>
          ) : (
            <button
              onClick={onPause}
              className="flex items-center space-x-1.5 px-4 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold shadow-md shadow-amber-600/30 transition-all"
            >
              <Pause className="w-4 h-4 fill-current" />
              <span>PAUSE</span>
            </button>
          )}

          <button
            onClick={onStop}
            disabled={!status.is_streaming && status.elapsed_sec === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 disabled:opacity-40 transition-all"
          >
            <Square className="w-3.5 h-3.5 fill-current" />
            <span>STOP</span>
          </button>

          <button
            onClick={() => {
              onStop();
              setTimeout(onPlay, 200);
            }}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all"
            title="Restart Stream from 00:00"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Right: Playback Speed selector */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">Speed:</span>
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
            {speeds.map((s) => (
              <button
                key={s}
                onClick={() => onSpeedChange(s)}
                className={`px-2 py-0.5 rounded transition-all ${
                  status.playback_speed === s
                    ? 'bg-slate-700 text-emerald-300 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline Scrub Progress Bar */}
      <div className="mt-3 pt-3 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
          <span>{formatTime(status.elapsed_sec)}</span>
          <span className="text-slate-500">
            Window Count: <strong className="text-slate-300">{status.window_count}</strong> (10s
            sliding inference)
          </span>
          <span>{formatTime(status.total_duration_sec)}</span>
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full transition-all duration-300"
            style={{ width: `${Math.min(100, Math.max(0, progressPct))}%` }}
          />
        </div>
      </div>
    </div>
  );
};
