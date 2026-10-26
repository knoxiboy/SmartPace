import React from 'react';
import { ArrhythmiaClass, CLASS_LABELS, CLASS_COLORS, PredictionData } from '../types';
import { ShieldCheck, AlertTriangle, Info } from 'lucide-react';

interface RhythmPredictionCardProps {
  prediction?: PredictionData;
}

export const RhythmPredictionCard: React.FC<RhythmPredictionCardProps> = ({ prediction }) => {
  const currentClass: string = prediction?.predicted_class ?? 'NSR';
  const confidence = prediction?.confidence ?? 0.982;
  const entropy = prediction?.entropy ?? 0.12;
  const uncertainty = prediction?.uncertainty_score ?? 0.08;

  const probabilities: Record<string, number> = prediction?.probabilities ?? {
    NSR: 0.982,
    SBRAD: 0.008,
    STACH: 0.004,
    AF: 0.003,
    PVC: 0.003,
  };

  const isArrhythmia = currentClass !== 'N' && currentClass !== 'NSR';
  const labelText = CLASS_LABELS[currentClass] || currentClass;
  const badgeColor = CLASS_COLORS[currentClass] || '#10b981';

  const displayClasses = Object.keys(probabilities).length > 0
    ? Object.keys(probabilities)
    : ['NSR', 'SBRAD', 'STACH', 'AF', 'PVC'];

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col justify-between">
      {/* Top Header */}
      <div>
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80 mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Rhythm Classification
            </span>
            <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 font-mono">
              ResNet-1D (AAMI / 5-Class)
            </span>
          </div>

          <div className="flex items-center space-x-1.5">
            <span className="text-[11px] text-slate-400 font-mono">T-Scaled:</span>
            <span className="text-[11px] text-emerald-400 font-mono font-semibold">T=1.42</span>
          </div>
        </div>

        {/* Classification Primary Banner */}
        <div
          className={`p-3 rounded-lg border flex items-center justify-between mb-4 ${
            isArrhythmia
              ? 'bg-rose-950/20 border-rose-500/40 text-rose-200'
              : 'bg-emerald-950/20 border-emerald-500/40 text-emerald-200'
          }`}
        >
          <div className="flex items-center space-x-3">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center font-bold font-mono text-xl text-white shadow-md"
              style={{ backgroundColor: badgeColor }}
            >
              {currentClass}
            </div>
            <div>
              <div className="font-semibold text-sm sm:text-base text-white tracking-tight">
                {labelText}
              </div>
              <div className="text-[11px] text-slate-400 font-mono">
                {isArrhythmia ? 'Pathological Ectopic Arrhythmia' : 'Physiological Sinus Conduction'}
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-2xl font-black font-mono text-white">
              {(confidence * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-slate-400 uppercase font-mono">Confidence</div>
          </div>
        </div>

        {/* 5-Class Probability Distribution Bars */}
        <div className="space-y-2 mb-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>5-Class Posterior Distribution</span>
            <span className="font-mono text-[10px]">Softmax Posterior</span>
          </div>

          {displayClasses.map((cls) => {
            const prob = probabilities[cls] ?? 0;
            const pct = (prob * 100).toFixed(1);
            const isTarget = cls === currentClass;
            const classLabel = CLASS_LABELS[cls] || cls;
            const classColor = CLASS_COLORS[cls] || '#38bdf8';

            return (
              <div key={cls} className="space-y-0.5">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="flex items-center space-x-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: classColor }}
                    />
                    <span className={isTarget ? 'text-white font-bold' : 'text-slate-400'}>
                      {cls} ({classLabel.split(' ')[0]})
                    </span>
                  </span>
                  <span className={isTarget ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                    {pct}%
                  </span>
                </div>
                <div className="w-full bg-slate-800/80 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.max(1, prob * 100)}%`,
                      backgroundColor: classColor,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Uncertainty & Entropy Telemetry */}
      <div className="pt-3 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
          <div className="text-[10px] text-slate-500 uppercase">Shannon Entropy</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">
            {entropy.toFixed(3)} <span className="text-[10px] text-slate-500">nats</span>
          </div>
          <div className="text-[10px] text-emerald-400">
            {entropy < 0.3 ? 'Low Dispersal' : 'High Ambiguity'}
          </div>
        </div>

        <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
          <div className="text-[10px] text-slate-500 uppercase">MC Uncertainty</div>
          <div className="text-sm font-bold text-slate-200 mt-0.5">
            {uncertainty.toFixed(3)}
          </div>
          <div className="text-[10px] text-sky-400">
            {uncertainty < 0.2 ? 'Deterministic' : 'Epistemic Spread'}
          </div>
        </div>
      </div>
    </div>
  );
};
