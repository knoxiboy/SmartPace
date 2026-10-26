import React, { useState, useEffect } from 'react';
import { Cpu, Zap, Activity, ShieldCheck, Play, RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import { BenchmarkResult } from '../types';

export const EdgeBenchmarkPanel: React.FC = () => {
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch latest benchmark result on mount
    api.getLatestBenchmark()
      .then((data) => {
        if (data && data.status === 'completed') {
          setBenchmark(data);
        }
      })
      .catch(() => {});
  }, []);

  const handleRunBenchmark = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await api.runBenchmark({ n_windows: 50, repetitions: 3, warmup_runs: 10 });
      setBenchmark(res);
    } catch (err: any) {
      setError(err?.message || 'Benchmark execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              PyTorch FP32 vs ONNX INT8 Benchmark
              <span className="px-2 py-0.5 text-[10px] bg-slate-800 text-slate-400 rounded-full border border-slate-700">
                Monotonic Monitored
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Measured model-only inference latency & quantization drift on identical ECG windows
            </p>
          </div>
        </div>

        <button
          onClick={handleRunBenchmark}
          disabled={isRunning}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isRunning
              ? 'bg-indigo-600/50 text-indigo-200 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/20 active:scale-95'
          }`}
        >
          {isRunning ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Benchmarking (50 Windows)...
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              Run 50-Window Benchmark
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg text-xs text-rose-400">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {benchmark ? (
        <div className="space-y-3">
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400">Median Speedup</span>
              <div className="text-lg font-bold text-emerald-400 flex items-baseline gap-1 mt-0.5">
                {benchmark.comparison.speedup_ratio_median.toFixed(1)}x
                <span className="text-[10px] text-slate-500 font-normal">faster</span>
              </div>
              <span className="text-[10px] text-slate-500">
                Mean: {benchmark.comparison.speedup_ratio_mean.toFixed(1)}x
              </span>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400">Prediction Agreement</span>
              <div className="text-lg font-bold text-indigo-400 mt-0.5">
                {benchmark.comparison.prediction_agreement_pct.toFixed(1)}%
              </div>
              <span className="text-[10px] text-slate-500">
                {benchmark.comparison.matching_windows}/{benchmark.test_windows_evaluated} identical
              </span>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400">Probability Drift (MAE)</span>
              <div className="text-lg font-bold text-cyan-400 mt-0.5">
                {(benchmark.comparison.mean_probability_drift * 100).toFixed(3)}%
              </div>
              <span className="text-[10px] text-slate-500">
                Max: {(benchmark.comparison.max_probability_drift * 100).toFixed(2)}%
              </span>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400">Model Compression</span>
              <div className="text-lg font-bold text-amber-400 mt-0.5">
                {benchmark.comparison.compression_ratio.toFixed(1)}x
              </div>
              <span className="text-[10px] text-slate-500">
                {benchmark.pytorch_fp32.size_kb} KB → {benchmark.onnx_int8.size_kb} KB
              </span>
            </div>
          </div>

          {/* Model Comparison Table */}
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3 font-medium">Model Variant</th>
                  <th className="py-2 px-3 font-medium">Runtime Provider</th>
                  <th className="py-2 px-3 font-medium">Median Latency</th>
                  <th className="py-2 px-3 font-medium">Mean Latency</th>
                  <th className="py-2 px-3 font-medium">95th Percentile (p95)</th>
                  <th className="py-2 px-3 font-medium">Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                <tr className="hover:bg-slate-800/30">
                  <td className="py-2 px-3 font-mono text-slate-200 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                    PyTorch FP32
                  </td>
                  <td className="py-2 px-3 text-slate-400 font-mono">
                    Device: {benchmark.pytorch_fp32.device.toUpperCase()}
                  </td>
                  <td className="py-2 px-3 font-semibold text-slate-200">
                    {benchmark.pytorch_fp32.median_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">
                    {benchmark.pytorch_fp32.mean_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">
                    {benchmark.pytorch_fp32.p95_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">{benchmark.pytorch_fp32.size_kb} KB</td>
                </tr>

                <tr className="hover:bg-slate-800/30">
                  <td className="py-2 px-3 font-mono text-emerald-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                    ONNX INT8 Quantized
                  </td>
                  <td className="py-2 px-3 text-slate-400 font-mono">
                    {benchmark.onnx_int8.execution_provider}
                  </td>
                  <td className="py-2 px-3 font-bold text-emerald-400">
                    {benchmark.onnx_int8.median_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">
                    {benchmark.onnx_int8.mean_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">
                    {benchmark.onnx_int8.p95_latency_ms.toFixed(2)} ms
                  </td>
                  <td className="py-2 px-3 text-slate-400">{benchmark.onnx_int8.size_kb} KB</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
          Click "Run 50-Window Benchmark" to measure high-resolution latency and quantization fidelity.
        </div>
      )}
    </div>
  );
};
