import React from 'react';
import {
  Activity,
  Cpu,
  ShieldAlert,
  Search,
  Sliders,
  BarChart3,
  Wifi,
  WifiOff,
  AlertTriangle
} from 'lucide-react';

export type ActiveTab = 'monitor' | 'explainability' | 'stress' | 'evaluation' | 'system';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  isWsConnected: boolean;
  isStreaming: boolean;
  recordName: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isWsConnected,
  isStreaming,
  recordName,
}) => {
  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode }[] = [
    { id: 'monitor', label: 'Live Monitor', icon: <Activity className="w-4 h-4" /> },
    { id: 'explainability', label: 'Explainability Lab', icon: <Search className="w-4 h-4" /> },
    { id: 'stress', label: 'Reliability & Stress', icon: <Sliders className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Research Evaluation', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'system', label: 'Model & System Info', icon: <Cpu className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0c101a]/95 backdrop-blur border-b border-slate-800">
      {/* Top Academic Research Disclaimer Banner */}
      <div className="bg-gradient-to-r from-amber-950/60 via-slate-900 to-amber-950/60 border-b border-amber-500/20 px-4 py-1.5 flex items-center justify-between text-xs text-amber-200">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="font-semibold tracking-wide">RESEARCH PROTOTYPE:</span>
          <span className="text-amber-200/90 hidden sm:inline">
            SmartPace Live is an academic demonstration platform for uncertainty-aware ECG reliability and explainability. Not clinically certified for medical diagnosis or pacemaker actuation.
          </span>
        </div>
        <div className="flex items-center space-x-3 shrink-0">
          <span className="text-[11px] bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded text-amber-300 font-mono">
            MIT-BIH + NSTDB
          </span>
          <span className="text-[11px] bg-slate-800 border border-slate-700 px-2 py-0.5 rounded text-slate-300 font-mono">
            v2.4.0-Live
          </span>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Brand Logo */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('monitor')}>
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm shadow-emerald-500/20">
              <Activity className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-base font-bold tracking-tight text-white">SmartPace</span>
                <span className="text-xs font-semibold px-1.5 py-0.2 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded">
                  LIVE
                </span>
              </div>
              <p className="text-[10px] text-slate-400 -mt-0.5 font-mono">Reliability AI & Explainability</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-slate-800 text-emerald-400 border border-slate-700 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <span className={active ? 'text-emerald-400' : 'text-slate-400'}>{item.icon}</span>
                  <span className="hidden md:inline">{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Real-time Status Badges */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* Record Pill */}
            <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
              <span className="text-slate-500">REC:</span>
              <span className="text-cyan-400 font-semibold">{recordName || 'None'}</span>
            </div>

            {/* Stream State */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-medium border ${
                isStreaming
                  ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-400'
                  : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isStreaming ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'
                }`}
              />
              <span className="text-[11px] font-mono">{isStreaming ? 'STREAMING' : 'IDLE'}</span>
            </div>

            {/* WebSocket Connection Pill */}
            <div
              className={`flex items-center space-x-1 px-2 py-1 rounded text-xs border ${
                isWsConnected
                  ? 'bg-emerald-950/30 border-emerald-500/20 text-emerald-400'
                  : 'bg-rose-950/30 border-rose-500/20 text-rose-400'
              }`}
              title={isWsConnected ? 'WebSocket Telemetry Connected' : 'WebSocket Disconnected'}
            >
              {isWsConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
              <span className="text-[11px] font-mono hidden sm:inline">
                {isWsConnected ? 'WS-OK' : 'DISCONNECTED'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
