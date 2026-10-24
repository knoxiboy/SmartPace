import React, { useState, useEffect, useRef } from 'react';
import {
  Search,
  Sparkles,
  Sliders,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  Cpu,
  Layers,
  Activity,
  ArrowRight
} from 'lucide-react';
import { api } from '../services/api';
import {
  EventItem,
  ExplainResponse,
  StressTestResponse,
  FailureCase,
  ArrhythmiaClass,
  CLASS_LABELS,
  CLASS_COLORS
} from '../types';

interface ExplainabilityLabPageProps {
  selectedEventForExplanation?: EventItem | null;
}

export const ExplainabilityLabPage: React.FC<ExplainabilityLabPageProps> = ({
  selectedEventForExplanation,
}) => {
  const [targetClass, setTargetClass] = useState<string>('auto');
  const [nSteps, setNSteps] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  // Stress test state
  const [stressLoading, setStressLoading] = useState<boolean>(false);
  const [stressResult, setStressResult] = useState<StressTestResponse | null>(null);
  const [stressNoise, setStressNoise] = useState<string>('em');
  const [stressSnr, setStressSnr] = useState<number>(12);

  // Failure cases state
  const [failureCases, setFailureCases] = useState<FailureCase[]>([]);
  const [selectedCase, setSelectedCase] = useState<FailureCase | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load failure cases on mount
  useEffect(() => {
    api.getFailureCases().then((res) => {
      setFailureCases(res.cases || []);
      if (res.cases && res.cases.length > 0) {
        setSelectedCase(res.cases[0]);
      }
    }).catch(console.error);
  }, []);

  // Compute explanation when event or case changes
  useEffect(() => {
    runComputeExplanation();
  }, [selectedEventForExplanation, selectedCase, targetClass, nSteps]);

  const runComputeExplanation = async () => {
    setLoading(true);
    try {
      const params: any = {
        n_steps: nSteps,
      };

      if (targetClass !== 'auto') {
        params.target_class = targetClass;
      }

      if (selectedEventForExplanation?.id) {
        params.event_id = selectedEventForExplanation.id;
      } else if (selectedCase?.signal_preview) {
        params.signal = selectedCase.signal_preview;
      }

      const res = await api.getExplanation(params);
      setExplanation(res);
      setStressResult(null); // reset stress test when new explanation loads
    } catch (err) {
      console.error('Explanation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunStressTest = async () => {
    if (!explanation) return;
    setStressLoading(true);
    try {
      const res = await api.runStressTest({
        event_id: selectedEventForExplanation?.id,
        signal: explanation.clean_signal,
        noise_kind: stressNoise,
        snr_db: stressSnr,
      });
      setStressResult(res);
    } catch (err) {
      console.error('Stress test error:', err);
    } finally {
      setStressLoading(false);
    }
  };

  // Render ECG Waveform with Integrated Gradients attribution overlay on Canvas
  useEffect(() => {
    if (!canvasRef.current || !explanation || !explanation.clean_signal) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = '#111827';
    for (let x = 0; x < width; x += 15) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 15) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const signal = explanation.clean_signal;
    const attributions = explanation.attributions || [];
    const N = signal.length;
    if (N < 2) return;

    const centerY = height / 2;
    const dx = width / (N - 1);

    // Find attribution max for normalized coloring
    const maxAttr = Math.max(...attributions.map((a) => Math.abs(a)), 1e-6);

    // Draw Heatmap bars underneath waveform
    for (let i = 0; i < N; i++) {
      const attr = attributions[i] || 0;
      const norm = Math.min(1, Math.abs(attr) / maxAttr);

      if (norm > 0.05) {
        // Positive attribution (supports predicted class): Emerald/Cyan; Negative: Rose
        ctx.fillStyle =
          attr >= 0
            ? `rgba(16, 185, 129, ${norm * 0.45})`
            : `rgba(244, 63, 94, ${norm * 0.45})`;
        ctx.fillRect(i * dx - dx * 0.5, 0, Math.max(1, dx), height);
      }
    }

    // Baseline
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw ECG Signal line
    ctx.lineWidth = 2.0;
    ctx.strokeStyle = '#38bdf8'; // cyan trace
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = i * dx;
      const y = centerY - signal[i] * (height * 0.35);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Mark QRS / anatomical boundaries
    if (explanation.anatomical_regions) {
      explanation.anatomical_regions.forEach((region) => {
        const [startIdx, endIdx] = region.indices;
        const x1 = startIdx * dx;
        const x2 = endIdx * dx;

        ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(x1, 10, x2 - x1, height - 20);
        ctx.setLineDash([]);

        ctx.fillStyle = '#c084fc';
        ctx.font = 'bold 9px monospace';
        ctx.fillText(
          `${region.name} (${(region.relative_importance * 100).toFixed(0)}%)`,
          x1 + 4,
          24
        );
      });
    }
  }, [explanation]);

  return (
    <div className="space-y-4">
      {/* Header Banner */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Search className="w-5 h-5 text-emerald-400" />
            <h1 className="text-base font-bold text-white tracking-tight">
              Explainability Lab: Axiomatic Integrated Gradients
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Path-integrated attribution over Gauss-Legendre quadrature with anatomical QRS / P-wave / ST localization.
          </p>
        </div>

        {/* Algorithm Configuration */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-mono">
            <span className="text-slate-400">Target:</span>
            <select
              value={targetClass}
              onChange={(e) => setTargetClass(e.target.value)}
              className="bg-transparent text-emerald-400 font-bold focus:outline-none"
            >
              <option value="auto">Auto (Predicted)</option>
              {(['N', 'S', 'V', 'F', 'Q'] as ArrhythmiaClass[]).map((c) => (
                <option key={c} value={c}>
                  Class {c} ({CLASS_LABELS[c].split(' ')[0]})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-mono">
            <span className="text-slate-400">Steps (m):</span>
            <select
              value={nSteps}
              onChange={(e) => setNSteps(Number(e.target.value))}
              className="bg-transparent text-cyan-400 font-bold focus:outline-none"
            >
              <option value={20}>20 steps</option>
              <option value={30}>30 steps (Default)</option>
              <option value={50}>50 steps (High Res)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Attribution Display Canvas */}
      <div className="rounded-xl border border-slate-800 bg-[#0a0d16] p-4 shadow-lg">
        <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-xs font-mono">
          <div className="flex items-center space-x-2">
            <span className="text-slate-300 font-semibold uppercase">
              Attribution-Waveform Overlay
            </span>
            <span className="text-slate-500">•</span>
            <span className="text-emerald-400">Green = Positive Gradient</span>
            <span className="text-slate-500">•</span>
            <span className="text-rose-400">Red = Negative Gradient</span>
          </div>
          <span className="text-slate-500">
            Baseline: <strong className="text-slate-300">Zero-voltage reference</strong>
          </span>
        </div>

        <div className="relative w-full h-72 rounded-lg overflow-hidden bg-[#080c14]">
          <canvas ref={canvasRef} className="w-full h-full block" />
          {loading && (
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center text-xs font-mono text-emerald-300">
              Integrating Gradients over path (m={nSteps})...
            </div>
          )}
        </div>

        {/* Anatomical Regions Importance Bar */}
        {explanation?.anatomical_regions && (
          <div className="mt-4 pt-3 border-t border-slate-800">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
              Anatomical Segment Attribution Breakdown
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {explanation.anatomical_regions.map((region) => (
                <div
                  key={region.name}
                  className="p-2.5 rounded-lg border border-slate-800 bg-slate-900/60"
                >
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-xs font-bold text-white">{region.name}</span>
                    <span className="text-xs font-mono font-bold text-purple-400">
                      {(region.relative_importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{ width: `${Math.min(100, region.relative_importance * 100)}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-400 leading-tight">
                    {region.clinical_significance}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Two Column Grid: Attribution Stability Stress Test & Case C Silent-Failure Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Module 1: Attribution Stability Test (Spearman Rank Correlation rho) */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-3">
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                  Perturbation Stability Test (Spearman ρ)
                </span>
              </div>
              <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 font-mono">
                Attribution Invariance
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Evaluates if Integrated Gradients attributions remain stable under non-semantic noise perturbation, verifying that the explanation does not collapse or hallucinate.
            </p>

            <div className="flex flex-wrap items-center gap-2 mb-4 text-xs font-mono">
              <label className="text-slate-400">Noise:</label>
              <select
                value={stressNoise}
                onChange={(e) => setStressNoise(e.target.value)}
                className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1"
              >
                <option value="em">Electrode Motion (EM)</option>
                <option value="bw">Baseline Wander (BW)</option>
                <option value="ma">Muscle Artifact (MA)</option>
              </select>

              <label className="text-slate-400 ml-2">SNR (dB):</label>
              <select
                value={stressSnr}
                onChange={(e) => setStressSnr(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1"
              >
                <option value={18}>18 dB (Mild)</option>
                <option value={12}>12 dB (Moderate)</option>
                <option value={6}>6 dB (Severe)</option>
              </select>

              <button
                onClick={handleRunStressTest}
                disabled={stressLoading || !explanation}
                className="ml-auto px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs transition-colors disabled:opacity-40"
              >
                {stressLoading ? 'Computing...' : 'Run Stability Test'}
              </button>
            </div>

            {stressResult && (
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-2 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Attribution Rank Stability (ρ):</span>
                  <span
                    className={`text-base font-bold ${
                      stressResult.attribution_stability_rho >= 0.7
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }`}
                  >
                    ρ = {stressResult.attribution_stability_rho.toFixed(3)}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Counterfactual Class Flip:</span>
                  <span
                    className={`font-bold ${
                      stressResult.counterfactual_flips ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {stressResult.counterfactual_flips ? 'FLIPPED (Unstable)' : 'PRESERVED (Robust)'}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-400">QRS Focus Drop:</span>
                  <span className="text-slate-200">
                    {stressResult.anatomical_shift.qrs_focus_drop_pct.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Module 2: Case C Silent-Failure Mode Inspector */}
        <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-3">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                  Case C Silent-Failure Inspection
                </span>
              </div>
              <span className="text-[10px] bg-purple-950/60 text-purple-300 border border-purple-800 px-1.5 py-0.5 rounded font-mono">
                OOD / Shift Failure Mode
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Case C represents the most dangerous medical ML vulnerability: <em>clean signal, high softmax confidence, but incorrect classification due to latent distribution shift</em>.
            </p>

            {/* Select Failure Case */}
            <div className="space-y-2 mb-3">
              {failureCases.map((fc) => (
                <div
                  key={fc.case_id}
                  onClick={() => setSelectedCase(fc)}
                  className={`p-2.5 rounded-lg border text-xs cursor-pointer transition-all ${
                    selectedCase?.case_id === fc.case_id
                      ? 'bg-purple-950/30 border-purple-500/50 text-white'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-bold text-slate-200">{fc.title}</span>
                    <span className="text-[10px] font-mono text-purple-400">{fc.case_id}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2">{fc.description}</p>
                </div>
              ))}
            </div>

            {selectedCase && (
              <div className="p-3 rounded-lg bg-slate-900/90 border border-purple-500/30 text-xs font-mono space-y-1.5">
                <div className="text-emerald-400 font-semibold">
                  Why Conventional Softmax Fails:
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  The model assigns &gt;95% confidence because the test beat closely resembles training features in Euclidean projection, but is physiologically anomalous.
                </p>
                <div className="text-purple-400 font-semibold pt-1">
                  How SmartPace Intercepts:
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  Mahalanobis latent shift score ({selectedCase.shift_score.toFixed(1)}) exceeds threshold λ=18.5, causing Gate 3 to issue an <strong>ABSTAIN_UNFAMILIAR</strong> veto, preventing hazardous misdiagnosis!
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
