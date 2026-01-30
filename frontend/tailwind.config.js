
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'matrix-black': '#050505',
        'matrix-dark': '#0a0f0a',
        'matrix-card': '#0a120a',
        'matrix-border': '#1a2f1a',
        'neon-green': '#00ff6a',
        'neon-green-dim': '#00cc55',
        'neon-green-dark': '#004422',
        'emerald': '#10b981',
        'lime': '#84cc16',
        'forest': '#0d140d',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'pulse-green 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
