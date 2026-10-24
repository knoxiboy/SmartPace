import React, { useRef, useEffect, useState, useMemo } from 'react';
import { ZoomIn, ZoomOut, Pause, Play, Sparkles } from 'lucide-react';

interface ECGWaveformCanvasProps {
  samples: { value: number; is_rpeak: boolean; timestamp: number }[];
  currentRecord: string;
  leadName?: string;
  samplingRate?: number;
  highlightWindowStart?: number;
}

export const ECGWaveformCanvas: React.FC<ECGWaveformCanvasProps> = ({
  samples,
  currentRecord,
  leadName = 'MLII',
  samplingRate = 250,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [gain, setGain] = useState<number>(1.0);
  const [windowDurationSec, setWindowDurationSec] = useState<number>(6.0); // 6 seconds on screen
  const [isFrozen, setIsFrozen] = useState<boolean>(false);
  const [frozenSamples, setFrozenSamples] = useState<typeof samples>([]);

  // When user toggles freeze, hold the current sample slice
  const handleToggleFreeze = () => {
    if (!isFrozen) {
      setFrozenSamples([...samples]);
      setIsFrozen(true);
    } else {
      setIsFrozen(false);
    }
  };

  const activeSamples = isFrozen ? frozenSamples : samples;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear background
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, width, height);

    // Draw ECG Medical Grid
    // Major grid lines: every 50px (simulating 0.2s / 5mm)
    // Minor grid lines: every 10px (simulating 0.04s / 1mm)
    const minorStep = 12;
    const majorStep = minorStep * 5;

    ctx.lineWidth = 0.5;
    ctx.strokeStyle = '#111827';
    ctx.beginPath();
    for (let x = 0; x < width; x += minorStep) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
    }
    for (let y = 0; y < height; y += minorStep) {
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
    }
    ctx.stroke();

    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1e293b';
    ctx.beginPath();
    for (let x = 0; x < width; x += majorStep) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
    }
    for (let y = 0; y < height; y += majorStep) {
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
    }
    ctx.stroke();

    // Center Baseline
    const centerY = height / 2;
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.9)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Calculate how many samples to display based on window duration
    const maxVisibleSamples = Math.floor(windowDurationSec * samplingRate);
    const startIndex = Math.max(0, activeSamples.length - maxVisibleSamples);
    const visibleData = activeSamples.slice(startIndex);

    if (visibleData.length < 2) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px monospace';
      ctx.fillText('Awaiting ECG signal telemetry stream...', 20, centerY);
      return;
    }

    const dx = width / (maxVisibleSamples - 1);
    const amplitudeScale = (height * 0.35) * gain;

    // Draw Waveform Glow & Line
    ctx.shadowColor = 'rgba(16, 185, 129, 0.45)';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.0;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    for (let i = 0; i < visibleData.length; i++) {
      const x = i * dx;
      // Invert y because canvas (0,0) is top-left
      const y = centerY - visibleData[i].value * amplitudeScale;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Reset shadow
    ctx.shadowBlur = 0;

    // Draw Detected R-Peaks
    for (let i = 0; i < visibleData.length; i++) {
      if (visibleData[i].is_rpeak) {
        const x = i * dx;
        const y = centerY - visibleData[i].value * amplitudeScale;

        // Draw inverted peak marker
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Marker label
        ctx.fillStyle = '#fecdd3';
        ctx.font = 'bold 9px monospace';
        ctx.fillText('R', x - 3, y - 8);

        // Small drop line
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + 14);
        ctx.stroke();
      }
    }

    // Lead & Scale Overlay
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px monospace';
    ctx.fillText(`${leadName} • 25 mm/s • 10 mm/mV (x${gain.toFixed(1)})`, 14, 20);

    // Pulse sweep indicator at current head
    if (!isFrozen && visibleData.length > 0) {
      const headX = (visibleData.length - 1) * dx;
      const headY = centerY - visibleData[visibleData.length - 1].value * amplitudeScale;

      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(headX, headY, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = 'rgba(52, 211, 153, 0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(headX, headY, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [activeSamples, gain, windowDurationSec, isFrozen, samplingRate, leadName]);

  return (
    <div className="relative rounded-xl border border-slate-800 bg-[#0a0d16] p-3 shadow-lg shadow-black/40">
      {/* Top Banner controls */}
      <div className="flex flex-wrap items-center justify-between mb-2 pb-2 border-b border-slate-800/80 gap-2">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Real-Time Lead {leadName} ({samplingRate} Hz)
          </span>
          <span className="text-xs text-slate-500 font-mono">
            Record: <strong className="text-cyan-400 font-mono">{currentRecord}</strong>
          </span>
        </div>

        {/* View Controls */}
        <div className="flex items-center space-x-2">
          {/* Time window selector */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[11px] font-mono">
            {[4, 6, 10].map((sec) => (
              <button
                key={sec}
                onClick={() => setWindowDurationSec(sec)}
                className={`px-2 py-0.5 rounded ${
                  windowDurationSec === sec
                    ? 'bg-slate-700 text-emerald-300 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {sec}s
              </button>
            ))}
          </div>

          {/* Gain selector */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-[11px] font-mono">
            <button
              onClick={() => setGain((prev) => Math.max(0.5, prev - 0.25))}
              className="p-1 text-slate-400 hover:text-slate-200"
              title="Decrease Amplitude Gain"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-1 text-slate-300">{gain.toFixed(1)}x</span>
            <button
              onClick={() => setGain((prev) => Math.min(3.0, prev + 0.25))}
              className="p-1 text-slate-400 hover:text-slate-200"
              title="Increase Amplitude Gain"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Freeze toggle */}
          <button
            onClick={handleToggleFreeze}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
              isFrozen
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700'
            }`}
          >
            {isFrozen ? (
              <>
                <Play className="w-3 h-3 fill-current" />
                <span>UNFREEZE</span>
              </>
            ) : (
              <>
                <Pause className="w-3 h-3 fill-current" />
                <span>FREEZE</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Canvas Element */}
      <div ref={containerRef} className="relative w-full h-72 sm:h-80 overflow-hidden rounded-lg">
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />

        {isFrozen && (
          <div className="absolute top-3 right-3 bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs px-2.5 py-1 rounded font-mono font-semibold backdrop-blur">
            WAVEFORM FROZEN
          </div>
        )}
      </div>

      {/* Bottom telemetry footer */}
      <div className="flex items-center justify-between mt-2 pt-1 text-[11px] font-mono text-slate-400">
        <div>
          <span>BUFFER: {activeSamples.length} samples</span>
          <span className="mx-2">•</span>
          <span>WINDOW: {windowDurationSec}s span</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-slate-500">Pan-Tompkins QRS Detection Active</span>
          <span className="inline-block w-2 h-2 rounded-full bg-rose-500" />
          <span className="text-slate-300">R-Peaks</span>
        </div>
      </div>
    </div>
  );
};
