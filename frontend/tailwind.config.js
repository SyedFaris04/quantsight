/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: { 300: "#c9c7c3", 400: "#a9a8a4", 500: "#92918d", 600: "#83827e", 700: "#3a3b3f", 800: "#26282c", 900: "#18191c", 950: "#111214" },
        white: "#ebe9e6",
        // Single interface accent (links, selection, focus). Named indigo so existing classes follow it.
        indigo: { 200: "#cfdffb", 300: "#a7c4f7", 400: "#86aef4", 500: "#6ea0f2", 600: "#4f86e4", 700: "#3a6cc4", 800: "#2d548f", 900: "#223f6a", 950: "#172a47" },
        // NuroQuant brand colours
        brand: {
          50:  "#eef2ff",
          100: "#e0e7ff",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
          900: "#1e1b4b",
        },
        buy:  "#16a34a",   // green-600
        sell: "#dc2626",   // red-600
        hold: "#d97706",   // amber-600
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
