import React from 'react';
import { DecisionData, DecisionStatus } from '../types';
import {
  ShieldCheck,
  ShieldX,
  AlertTriangle,
  HelpCircle,
  CheckCircle,
  XCircle,
  Info
} from 'lucide-react';

interface ReliabilityTriGateProps {
  decision?: DecisionData;
  sqiScore?: number;
  uncertaintyScore?: number;
  shiftScore?: number;
}

export const ReliabilityTriGate: React.FC<ReliabilityTriGateProps> = ({
  decision,
  sqiScore = 0.88,
  uncertaintyScore = 0.08,
  shiftScore = 6.4,
}) => {
  const status: DecisionStatus = decision?.status ?? 'ACCEPT';
  const reason = decision?.reason ?? 'Clean signal, low uncertainty, in-distribution morphology.';
  const gateFailures = decision?.gate_failures ?? [];

  // Gate 1: SQI
  const gate1Pass = !gateFailures.includes('signal_quality_sqi');
  // Gate 2: Uncertainty
  const gate2Pass = !gateFailures.includes('predictive_uncertainty');
  // Gate 3: Shift / OOD
  const gate3Pass = !gateFailures.includes('distribution_shift_mahalanobis');

  const getStatusBadge = (s: DecisionStatus) => {
    switch (s) {
      case 'ACCEPT':
        return {
          title: 'DECISION ACCEPTED',
          desc: 'High reliability. Pass verified on all 3 safety gates.',
          bg: 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300',
          icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
          pill: 'bg-emerald-500 text-slate-950',
        };
      case 'REJECT_QUALITY':
        return {
          title: 'REJECT: POOR SIGNAL QUALITY',
          desc: 'Waveform corrupted by noise, baseline wander, or electrode artifact.',
          bg: 'bg-amber-950/40 border-amber-500/50 text-amber-300',
          icon: <ShieldX className="w-5 h-5 text-amber-400" />,
          pill: 'bg-amber-500 text-slate-950',
        };
      case 'ABSTAIN_UNCERTAIN':
        return {
          title: 'ABSTAIN: PREDICTIVE UNCERTAINTY',
          desc: 'Model entropy exceeded tolerance. Beat rhythm pattern is ambiguous.',
          bg: 'bg-indigo-950/40 border-indigo-500/50 text-indigo-300',
          icon: <HelpCircle className="w-5 h-5 text-indigo-400" />,
          pill: 'bg-indigo-500 text-white',
        };
      case 'ABSTAIN_UNFAMILIAR':
        return {
          title: 'ABSTAIN: UNFAMILIAR MORPHOLOGY',
          desc: 'Latent feature shift detected (Mahalanobis distance > threshold). Unseen distribution.',
          bg: 'bg-purple-950/40 border-purple-500/50 text-purple-300',
          icon: <AlertTriangle className="w-5 h-5 text-purple-400" />,
          pill: 'bg-purple-500 text-white',
        };
    }
  };

  const badge = getStatusBadge(status);

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80 mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Reliability Tri-Gate Architecture
            </span>
            <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 font-mono">
              3-Stage Veto
            </span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Selective Abstention Policy</span>
        </div>

        {/* Primary Status Banner */}
        <div className={`p-3 rounded-lg border flex items-start space-x-3 mb-4 ${badge.bg}`}>
          <div className="p-1 rounded-md bg-slate-900/60 shrink-0 mt-0.5">{badge.icon}</div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm tracking-tight">{badge.title}</span>
              <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded font-mono ${badge.pill}`}>
                {status}
              </span>
            </div>
            <p className="text-xs mt-1 text-slate-300 leading-relaxed">{badge.desc}</p>
          </div>
        </div>

        {/* The 3 Gates Interactive Breakdown */}
        <div className="space-y-2.5 mb-3">
          {/* Gate 1: Signal Quality */}
          <div
            className={`p-2.5 rounded-lg border text-xs transition-colors ${
              gate1Pass
                ? 'bg-slate-900/40 border-slate-800 text-slate-300'
                : 'bg-rose-950/30 border-rose-500/40 text-rose-200'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {gate1Pass ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                )}
                <div>
                  <div className="font-semibold text-white">Gate 1: Signal Quality (SQI)</div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    Criteria: Composite SQI ≥ 0.50
                  </div>
                </div>
              </div>
              <div className="text-right font-mono">
                <div className={gate1Pass ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                  {(sqiScore * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-slate-500">{gate1Pass ? 'PASSED' : 'VETOED'}</div>
              </div>
            </div>
          </div>

          {/* Gate 2: Predictive Uncertainty */}
          <div
            className={`p-2.5 rounded-lg border text-xs transition-colors ${
              gate2Pass
                ? 'bg-slate-900/40 border-slate-800 text-slate-300'
                : 'bg-indigo-950/30 border-indigo-500/40 text-indigo-200'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {gate2Pass ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-4 h-4 text-indigo-400 shrink-0" />
                )}
                <div>
                  <div className="font-semibold text-white">Gate 2: Predictive Uncertainty</div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    Criteria: Epistemic Uncertainty ≤ τ = 0.45
                  </div>
                </div>
              </div>
              <div className="text-right font-mono">
                <div className={gate2Pass ? 'text-emerald-400 font-bold' : 'text-indigo-400 font-bold'}>
                  {uncertaintyScore.toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-500">{gate2Pass ? 'PASSED' : 'VETOED'}</div>
              </div>
            </div>
          </div>

          {/* Gate 3: Mahalanobis Latent Shift */}
          <div
            className={`p-2.5 rounded-lg border text-xs transition-colors ${
              gate3Pass
                ? 'bg-slate-900/40 border-slate-800 text-slate-300'
                : 'bg-purple-950/30 border-purple-500/40 text-purple-200'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {gate3Pass ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-4 h-4 text-purple-400 shrink-0" />
                )}
                <div>
                  <div className="font-semibold text-white">Gate 3: Shift / OOD Detection</div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    Criteria: Mahalanobis Distance ≤ λ = 18.5
                  </div>
                </div>
              </div>
              <div className="text-right font-mono">
                <div className={gate3Pass ? 'text-emerald-400 font-bold' : 'text-purple-400 font-bold'}>
                  {shiftScore.toFixed(1)}
                </div>
                <div className="text-[10px] text-slate-500">{gate3Pass ? 'PASSED' : 'VETOED'}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Diagnostic Recommendation */}
      <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
        <div className="flex items-center space-x-1.5 text-slate-400 mb-1">
          <Info className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold text-[11px] text-slate-300">
            System Diagnostic Reason:
          </span>
        </div>
        <p className="text-[11px] text-slate-300 leading-normal font-mono pl-5">
          {reason}
        </p>
      </div>
    </div>
  );
};
