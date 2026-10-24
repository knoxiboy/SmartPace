import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  CheckCircle,
  FileText,
  Activity,
  Layers,
  Cpu,
  Flame,
  Award,
  Download
} from 'lucide-react';
import { api } from '../services/api';

export const ResearchEvaluationPage: React.FC = () => {
  const [activeExp, setActiveExp] = useState<string>('exp1_architecture_comparison');
  const [experiments, setExperiments] = useState<any[]>([]);
  const [resultsData, setResultsData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    api.getExperiments().then((res) => {
      setExperiments(res.experiments || []);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    setLoading(true);
    api.getExperimentResults(activeExp)
      .then((res) => setResultsData(res))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [activeExp]);

  const expNav = [
    { id: 'exp1_architecture_comparison', label: 'Exp 1: Architecture Benchmarks' },
    { id: 'exp2_uncertainty_quantification', label: 'Exp 2: Uncertainty (UQ)' },
    { id: 'exp3_calibration', label: 'Exp 3: Temperature Scaling' },
    { id: 'exp4_noise_robustness', label: 'Exp 4: Noise Robustness (NSTDB)' },
    { id: 'exp5_selective_classification', label: 'Exp 5: Selective Abstention' },
    { id: 'exp6_edge_deployment', label: 'Exp 6: Edge & Embedded Latency' },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            <h1 className="text-base font-bold text-white tracking-tight">
              Research Evaluation: Experimental Suite (Exp 1 – Exp 6)
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Empirical validation results loaded directly from rigorous model benchmarks on MIT-BIH and NSTDB databases.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-[11px] bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-2.5 py-1 rounded font-mono font-semibold">
            Inter-Patient Protocol DS1/DS2
          </span>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex overflow-x-auto space-x-1 p-1 bg-slate-900 border border-slate-800 rounded-xl text-xs font-mono">
        {expNav.map((exp) => (
          <button
            key={exp.id}
            onClick={() => setActiveExp(exp.id)}
            className={`px-3 py-2 rounded-lg whitespace-nowrap transition-all ${
              activeExp === exp.id
                ? 'bg-slate-800 text-emerald-400 font-bold border border-slate-700 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            {exp.label}
          </button>
        ))}
      </div>

      {/* Results Container */}
      <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
        {loading ? (
          <div className="py-12 text-center text-xs font-mono text-slate-400">
            Loading experimental artifact data...
          </div>
        ) : (
          <div className="space-y-4">
            {/* Experiment Description & Research Question */}
            <div className="p-3.5 rounded-lg border border-slate-800 bg-slate-900/60">
              <div className="text-xs font-bold text-white mb-1">
                {resultsData?.metadata?.title || activeExp}
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {resultsData?.metadata?.description ||
                  'Evaluates model efficacy, reliability, and calibration across standardized patient divisions.'}
              </p>
              {resultsData?.metadata?.takeaway && (
                <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex items-start space-x-2 text-xs">
                  <Award className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="text-emerald-300">
                    <strong>Core Finding: </strong> {resultsData.metadata.takeaway}
                  </span>
                </div>
              )}
            </div>

            {/* Render Tabular Data */}
            {resultsData?.data && Array.isArray(resultsData.data) && resultsData.data.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 bg-slate-900/40">
                      {Object.keys(resultsData.data[0]).map((key) => (
                        <th key={key} className="py-2.5 px-3 uppercase tracking-wider">
                          {key.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {resultsData.data.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                        {Object.entries(row).map(([k, val]: any, j: number) => {
                          const isNumber = typeof val === 'number';
                          const displayVal = isNumber
                            ? Number.isInteger(val)
                              ? val
                              : val.toFixed(4)
                            : String(val);
                          const isPrimary = j === 0;

                          return (
                            <td
                              key={k}
                              className={`py-2.5 px-3 whitespace-nowrap ${
                                isPrimary
                                  ? 'text-white font-bold'
                                  : 'text-slate-300'
                              }`}
                            >
                              {displayVal}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-slate-400 text-xs font-mono py-4">
                Raw experiment data loaded successfully.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
