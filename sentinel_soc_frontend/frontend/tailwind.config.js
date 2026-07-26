/** 
 * tailwind.config.js
 * ===================
 * Design tokens for the SOC dashboard. Colors resolve through CSS custom
 * properties (see src/styles/index.css) so the whole palette can be
 * retuned in one place.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        void: 'var(--bg-void)',
        panel: 'var(--bg-panel)',
        'panel-raised': 'var(--bg-panel-raised)',
        hairline: 'var(--border-hairline)',
        ink: 'var(--text-ink)',
        'ink-dim': 'var(--text-ink-dim)',
        'ink-faint': 'var(--text-ink-faint)',
        signal: 'var(--accent-signal)',
        'signal-dim': 'var(--accent-signal-dim)',
        sev: {
          low: 'var(--sev-low)',
          medium: 'var(--sev-medium)',
          high: 'var(--sev-high)',
          critical: 'var(--sev-critical)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.02em' }],
      },
      borderRadius: {
        sm: '4px',
        DEFAULT: '6px',
        md: '8px',
      },
      boxShadow: {
        panel: '0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.6)',
      },
      keyframes: {
        'pulse-dot': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.35 },
        },
        'slide-in': {
          from: { transform: 'translateX(100%)' },
          to: { transform: 'translateX(0)' },
        },
        'row-in': {
          from: { opacity: 0, transform: 'translateY(-6px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        },
      },
      animation: {
        'pulse-dot': 'pulse-dot 1.6s ease-in-out infinite',
        'slide-in': 'slide-in 220ms cubic-bezier(0.16, 1, 0.3, 1)',
        'row-in': 'row-in 260ms ease-out',
      },
    },
  },
  plugins: [],
};
