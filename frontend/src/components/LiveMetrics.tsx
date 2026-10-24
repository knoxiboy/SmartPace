import React from 'react';
import { Heart, Activity, Gauge, Zap, CheckCircle2, AlertOctagon } from 'lucide-react';
import { VitalsData } from '../types';

interface LiveMetricsProps {
  vitals?: VitalsData;
  latencyMs?: number;
}

export const LiveMetrics: React.FC<LiveMetricsProps> = ({ vitals, latencyMs = 12 }) => {
  const hr = vitals?.heart_rate_bpm ?? 72;
  const rr = vitals?.rr_intervals_ms?.length
    ? Math.round(
        vitals.rr_intervals_ms.reduce((a, b) => a + b, 0) / vitals.rr_intervals_ms.length
      )
    : 833;

  const sqi = vitals?.sqi_scores?.composite_sqi ?? 0.88;
  const bwSqi = vitals?.sqi_scores?.baseline_wander_sqi ?? 0.94;
  const hfSqi = vitals?.sqi_scores?.high_freq_sqi ?? 0.91;
  const kurtSqi = vitals?.sqi_scores?.kurtosis_sqi ?? 0.86;

  // HR status classification
  const getHrColor = (rate: number) => {
    if (rate < 60) return { text: 'text-sky-400', label: 'Bradycardia', bg: 'bg-sky-500/10 border-sky-500/30' };
    if (rate > 100) return { text: 'text-rose-400', label: 'Tachycardia', bg: 'bg-rose-500/10 border-rose-500/30' };
    return { text: 'text-emerald-400', label: 'Normal Sinus', bg: 'bg-emerald-500/10 border-emerald-500/30' };
  };

  const hrStyle = getHrColor(hr);

  // Pulse animation duration based on HR
  const pulseDuration = hr > 0 ? `${(60 / hr).toFixed(2)}s` : '1s';

  // SQI quality label
  const getSqiStatus = (score: number) => {
    if (score >= 0.75) return { label: 'OPTIMAL QUALITY', color: 'text-emerald-400', bar: 'bg-emerald-500' };
    if (score >= 0.50) return { label: 'ACCEPTABLE', color: 'text-amber-400', bar: 'bg-amber-500' };
    return { label: 'POOR / NOISY', color: 'text-rose-400', bar: 'bg-rose-500' };
  };

  const sqiStatus = getSqiStatus(sqi);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* 1. Heart Rate (BPM) */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-3 shadow-md relative overflow-hidden">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span className="font-semibold uppercase tracking-wider">Heart Rate</span>
          <Heart
            className={`w-4 h-4 text-rose-500`}
            style={{ animation: `pulse ${pulseDuration} infinite ease-in-out` }}
          />
        </div>
        <div className="flex items-baseline space-x-1.5">
          <span className={`text-3xl font-extrabold font-mono tracking-tight ${hrStyle.text}`}>
            {hr > 0 ? hr : '--'}
          </span>
          <span className="text-xs text-slate-400 font-mono">BPM</span>
        </div>
        <div className="mt-2 flex items-center justify-between text-[11px]">
          <span className={`px-1.5 py-0.5 rounded border text-[10px] font-medium ${hrStyle.bg}`}>
            {hrStyle.label}
          </span>
          <span className="text-slate-500 font-mono">RR: {rr} ms</span>
        </div>
      </div>

      {/* 2. Mean RR Interval */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-3 shadow-md">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span className="font-semibold uppercase tracking-wider">RR Interval</span>
          <Activity className="w-4 h-4 text-sky-400" />
        </div>
        <div className="flex items-baseline space-x-1.5">
          <span className="text-3xl font-extrabold font-mono tracking-tight text-white">
            {rr}
          </span>
          <span className="text-xs text-slate-400 font-mono">ms</span>
        </div>
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
          <span>Target: 600-1000ms</span>
          <span className="text-emerald-400 font-mono">
            {rr >= 600 && rr <= 1000 ? 'Regular' : 'Irregular'}
          </span>
        </div>
      </div>

      {/* 3. Composite Signal Quality Index (SQI) */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-3 shadow-md">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span className="font-semibold uppercase tracking-wider">Signal Quality (SQI)</span>
          <Gauge className="w-4 h-4 text-emerald-400" />
        </div>
        <div className="flex items-baseline space-x-1.5">
          <span className={`text-3xl font-extrabold font-mono tracking-tight ${sqiStatus.color}`}>
            {(sqi * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-slate-400 font-mono">score</span>
        </div>
        {/* Progress bar */}
        <div className="mt-2 w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div
            className={`h-full ${sqiStatus.bar} transition-all duration-300`}
            style={{ width: `${Math.min(100, Math.max(0, sqi * 100))}%` }}
          />
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[10px] font-mono text-slate-400">
          <span className={sqiStatus.color}>{sqiStatus.label}</span>
          <span>BW: {(bwSqi * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* 4. Inference Latency & Pipeline State */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-3 shadow-md">
        <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
          <span className="font-semibold uppercase tracking-wider">Inference Latency</span>
          <Zap className="w-4 h-4 text-amber-400" />
        </div>
        <div className="flex items-baseline space-x-1.5">
          <span className="text-3xl font-extrabold font-mono tracking-tight text-amber-400">
            {latencyMs.toFixed(1)}
          </span>
          <span className="text-xs text-slate-400 font-mono">ms</span>
        </div>
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
          <span className="flex items-center space-x-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            <span>Real-Time Budget</span>
          </span>
          <span className="text-slate-500 font-mono">&lt; 40ms</span>
        </div>
      </div>
    </div>
  );
};
