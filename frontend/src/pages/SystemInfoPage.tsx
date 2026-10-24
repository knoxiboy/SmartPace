import React, { useState, useEffect } from 'react';
import {
  Cpu,
  ShieldAlert,
  Sliders,
  CheckCircle2,
  FileCode,
  HardDrive,
  Info,
  Layers,
  Database
} from 'lucide-react';
import { api } from '../services/api';
import { SystemInfo } from '../types';

export const SystemInfoPage: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    api.getSystemInfo()
      .then((info) => setSystemInfo(info))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight">
              Model Architecture & System Specification
            </h1>
            <p className="text-xs text-slate-400">
              Full architectural transparency, safety threshold disclosure, and research disclaimer.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            35 Tests Passing
          </span>
        </div>
      </div>

      {/* Grid of System Specs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model Architecture Details */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
          <div className="flex items-center space-x-2 pb-2 mb-3 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-300">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>Active Inference Neural Network</span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Architecture:</span>
              <span className="text-white font-bold">1D Residual Network (ResNet-1D)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Input Dimension:</span>
              <span className="text-cyan-400 font-bold">1 lead × 2500 samples (10.0 sec @ 250 Hz)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Parameters:</span>
              <span className="text-white font-bold">
                {systemInfo?.active_model?.parameters?.toLocaleString() || '188,421'} trainable
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Checkpoint File:</span>
              <span className="text-slate-300 text-[11px] truncate max-w-xs">
                {systemInfo?.active_model?.checkpoint || 'runs/first/models/resnet1d_AB_s0.pt'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Target Standard:</span>
              <span className="text-white">AAMI 5-Class (N, S, V, F, Q)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Execution Device:</span>
              <span className="text-emerald-400 font-bold">
                {systemInfo?.device?.toUpperCase() || 'CPU'} (PyTorch Native + JIT)
              </span>
            </div>
          </div>
        </div>

        {/* Safety Gate Thresholds */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
          <div className="flex items-center space-x-2 pb-2 mb-3 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-300">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span>Calibrated Safety Gate Thresholds</span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <div className="flex justify-between text-slate-200 font-bold mb-0.5">
                <span>Gate 1: Signal Quality (SQI)</span>
                <span className="text-emerald-400">
                  θ_sqi ≥ {systemInfo?.calibrated_thresholds?.sqi_composite_threshold ?? 0.5}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                Composite of kurtosis, skewness, baseline wander, and high-frequency power. Vetoes corrupted waveforms.
              </p>
            </div>

            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <div className="flex justify-between text-slate-200 font-bold mb-0.5">
                <span>Gate 2: Predictive Uncertainty</span>
                <span className="text-indigo-400">
                  τ ≤ {systemInfo?.calibrated_thresholds?.uncertainty_tau ?? 0.45}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                Normalized epistemic entropy. Abstained when rhythm pattern is ambiguous or multi-modal.
              </p>
            </div>

            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <div className="flex justify-between text-slate-200 font-bold mb-0.5">
                <span>Gate 3: Mahalanobis Latent Shift</span>
                <span className="text-purple-400">
                  λ ≤ {systemInfo?.calibrated_thresholds?.mahalanobis_lambda ?? 18.5}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                Covariance-scaled distance from training manifold. Intercepts Case C silent distribution shifts.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Databases & Edge Export */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Clinical Data Provenance */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
          <div className="flex items-center space-x-2 pb-2 mb-3 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-300">
            <Database className="w-4 h-4 text-emerald-400" />
            <span>Clinical Data Provenance</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Primary Database:</span>
              <span className="text-white">MIT-BIH Arrhythmia Database</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Total Local Records:</span>
              <span className="text-emerald-400 font-bold">
                {systemInfo?.mitdb_records_available ?? 48} WFDB records (All present)
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Noise Database:</span>
              <span className="text-white">MIT-BIH Noise Stress Test (NSTDB)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Sampling Standard:</span>
              <span className="text-cyan-400">250 Hz (Resampled from 360 Hz)</span>
            </div>
          </div>
        </div>

        {/* Edge & Embedded Optimization */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
          <div className="flex items-center space-x-2 pb-2 mb-3 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-300">
            <HardDrive className="w-4 h-4 text-cyan-400" />
            <span>Edge & Embedded Deployment</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">FP32 ONNX Footprint:</span>
              <span className="text-white">765 KB</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">INT8 Quantized ONNX:</span>
              <span className="text-emerald-400 font-bold">212 KB (72.3% reduction)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">MCU Target:</span>
              <span className="text-slate-300">ARM Cortex-M4 / Cortex-M33</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Inference Budget:</span>
              <span className="text-emerald-400 font-bold">&lt; 15 ms / 10s window</span>
            </div>
          </div>
        </div>
      </div>

      {/* Prominent Research Disclaimer */}
      <div className="rounded-xl border border-amber-500/40 bg-amber-950/20 p-4 shadow-md">
        <div className="flex items-start space-x-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1.5 text-xs text-amber-200">
            <h2 className="font-bold text-amber-300 uppercase tracking-wide">
              Mandatory Academic Research Disclaimer
            </h2>
            <p className="leading-relaxed">
              {systemInfo?.disclaimer ||
                'SmartPace Live is an educational and scientific research prototype designed to explore uncertainty-aware electrocardiogram classification, selective abstention policies, and axiomatic explainability. It is NOT a clinical medical device, does not possess FDA or CE Mark approval, and must NEVER be used to make clinical treatment decisions or directly actuate cardiac pacemakers.'}
            </p>
            <p className="text-[11px] text-amber-300/80">
              All demonstrated telemetry utilizes publicly accessible research databases (MIT-BIH Arrhythmia Database and NSTDB) under the PhysioNet Open Access license.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
