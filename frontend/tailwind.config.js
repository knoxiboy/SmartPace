/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ecg: {
          bg: '#0a0d14',
          card: '#111726',
          cardBorder: '#1e293b',
          grid: '#162235',
          gridSub: '#0d1522',
          line: '#10b981', // medical emerald green
          peak: '#f43f5e', // red r-peaks
          gold: '#f59e0b',
          blue: '#38bdf8',
          purple: '#a855f7',
          cyan: '#06b6d4',
          danger: '#ef4444',
          warning: '#f59e0b',
          success: '#10b981',
          abstain: '#6366f1',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
};
