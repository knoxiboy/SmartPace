import React, { useState } from 'react';
import { EventItem, ArrhythmiaClass, CLASS_COLORS, DecisionStatus } from '../types';
import { ExternalLink, Filter, ShieldCheck, AlertCircle, RefreshCw } from 'lucide-react';

interface EventTimelineProps {
  events: EventItem[];
  onSelectEventForExplanation: (event: EventItem) => void;
  onRefresh: () => void;
}

export const EventTimeline: React.FC<EventTimelineProps> = ({
  events,
  onSelectEventForExplanation,
  onRefresh,
}) => {
  const [filterStatus, setFilterStatus] = useState<string>('ALL');

  const filtered = events.filter((e) => {
    if (filterStatus === 'ALL') return true;
    return e.status === filterStatus;
  });

  const getStatusBadge = (status: DecisionStatus) => {
    switch (status) {
      case 'ACCEPT':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'REJECT_QUALITY':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'ABSTAIN_UNCERTAIN':
        return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40';
      case 'ABSTAIN_UNFAMILIAR':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
    }
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0f1422] p-4 shadow-md">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-2 mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Real-Time Inference Event Log
          </span>
          <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 font-mono">
            {events.length} Recorded
          </span>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center space-x-2">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-300 text-xs font-mono rounded px-2 py-1 focus:outline-none"
          >
            <option value="ALL">All Decisions ({events.length})</option>
            <option value="ACCEPT">Accepted Only</option>
            <option value="REJECT_QUALITY">Quality Rejected</option>
            <option value="ABSTAIN_UNCERTAIN">Uncertain Abstention</option>
            <option value="ABSTAIN_UNFAMILIAR">Unfamiliar Morphology</option>
          </select>

          <button
            onClick={onRefresh}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs border border-slate-700"
            title="Refresh Log"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Events Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-800/80 text-slate-400 bg-slate-900/40">
              <th className="py-2 px-2.5">TIME</th>
              <th className="py-2 px-2">WINDOW</th>
              <th className="py-2 px-2">REC</th>
              <th className="py-2 px-2">CLASS</th>
              <th className="py-2 px-2">CONF</th>
              <th className="py-2 px-2">SQI</th>
              <th className="py-2 px-2">UNCERT</th>
              <th className="py-2 px-2">SHIFT</th>
              <th className="py-2 px-2">STATUS</th>
              <th className="py-2 px-2 text-right">EXPLAIN</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-6 text-center text-slate-500">
                  No inference events matching current filter.
                </td>
              </tr>
            ) : (
              filtered.slice(0, 15).map((evt) => (
                <tr
                  key={evt.id}
                  className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                  onClick={() => onSelectEventForExplanation(evt)}
                >
                  <td className="py-2 px-2.5 text-slate-400 whitespace-nowrap">
                    {evt.timestamp.split('T')[1]?.slice(0, 8) || evt.timestamp.slice(0, 8)}
                  </td>
                  <td className="py-2 px-2 text-slate-300">
                    {(evt.window_start_sec ?? 0).toFixed(1)}s–{(evt.window_end_sec ?? 10).toFixed(1)}s
                  </td>
                  <td className="py-2 px-2 text-cyan-400 font-bold">{evt.record_name || 'MIT-BIH'}</td>
                  <td className="py-2 px-2">
                    <span
                      className="px-1.5 py-0.5 rounded font-bold text-white"
                      style={{ backgroundColor: CLASS_COLORS[evt.predicted_class] || '#10b981' }}
                    >
                      {evt.predicted_class}
                    </span>
                  </td>
                  <td className="py-2 px-2 font-bold text-white">
                    {((evt.confidence ?? 0.95) * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 px-2 text-slate-300">
                    {((evt.composite_sqi ?? 0.92) * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 px-2 text-slate-300">{(evt.uncertainty_score ?? 0.05).toFixed(3)}</td>
                  <td className="py-2 px-2 text-slate-300">{(evt.shift_score ?? 10.0).toFixed(1)}</td>
                  <td className="py-2 px-2">
                    <span
                      className={`px-1.5 py-0.5 rounded border text-[10px] font-bold ${getStatusBadge(
                        evt.status
                      )}`}
                    >
                      {evt.status}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEventForExplanation(evt);
                      }}
                      className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-emerald-600 hover:text-white text-emerald-400 border border-slate-700 text-[10px] transition-colors"
                      title="Inspect in Explainability Lab"
                    >
                      <span>Explain</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
