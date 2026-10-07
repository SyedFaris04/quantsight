const paths = {
  chat: "M3 4h18v13H9l-6 4z M7 8h10 M7 12h6",
  dashboard: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  market: "M3 20h18 M5 16l5-6 4 3 6-9",
  compare: "M6 3v18 M18 3v18 M3 7h6 M15 17h6 M10 12h4",
  results: "M5 4h14v17H5z M8 8h8 M8 12h8 M8 16h5",
  track: "M12 3a9 9 0 1 0 9 9 M12 7a5 5 0 1 0 5 5 M12 12l9-9 M16 3h5v5",
  backtest: "M3 3v18h18 M7 16l4-5 4 2 6-8",
  portfolio: "M3 7h18v14H3z M8 7V3h8v4 M3 12h18 M10 12v3h4v-3",
  learn: "M3 5l9-3 9 3v15l-9-3-9 3z M12 2v15",
  settings: "M4 7h16 M4 17h16 M8 4v6 M16 14v6",
  user: "M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0 M4 21v-3a8 8 0 0 1 16 0v3",
  menu: "M3 6h18 M3 12h18 M3 18h18",
  close: "M6 6l12 12 M18 6L6 18",
};
export default function Icon({ name, className = "w-5 h-5" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] || paths.results} />
    </svg>
  );
}
