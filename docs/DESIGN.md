# QuantSight interface contract

Read this before changing any page.

**Job of the interface.** People use it to find an instrument, read what the
models said, and check whether the evidence is good enough to believe. It
should read like an analyst's research report, not a marketing page.

**Audience.** A supervisor, a lecturer, students and new investors. They read
numbers carefully, so prefer tables to decoration.

## Tokens (`frontend/tailwind.config.js`, `frontend/src/index.css`)

### Type

| Use | Font | Size |
|---|---|---|
| Interface text | IBM Plex Sans | 14px body, 13px tables, 12px secondary |
| Tickers and code identifiers | IBM Plex Mono | — |
| Section titles | IBM Plex Sans | 18–20px |
| Page title | IBM Plex Sans | 28px |

- Weights are 400, 500 and 600.
- Numbers always use `tabular-nums`.

### Neutrals

Warm graphite, tinted rather than pure black or white:

| Role | Colour |
|---|---|
| Page | `gray-950` `#111214` |
| Surface | `gray-900` `#18191c` |
| Lines | `gray-800` `#26282c` |
| Text | `white` → `#ebe9e6` |

### Accent

One accent, blue `#6ea0f2` (the remapped `indigo` scale). It is used only for:
- links
- the selected state
- focus outlines
- the single primary button on a screen

### Meaning colours

| Colour | Meaning |
|---|---|
| Green | BUY / better |
| Red | SELL / worse |
| Amber | HOLD / caution |

Every one of them also carries a text label, never colour alone.

### Charts

- Model identity uses the fixed palette in `Compare.jsx` and `Detail.jsx`.
- New categorical palettes must pass the dataviz validator against `gray-900`.

### Shape

- Radius is 6px on surfaces and 4px on controls.
- 1px borders, no shadows, no gradients, no glass.

## Patterns

- **Lists of instruments are tables**, not card grids. Numbers are
  right-aligned, the header row is sticky where the table scrolls, and the
  ticker cell links to the analysis.
- **Summary figures are an inline strip of label and value**, not coloured
  tiles.
- **One level of container.** A surface may hold rules and spacing, never
  another bordered card.
- **Plain headers.** A page header is the title plus one sentence. There is no
  small uppercase label above the title.
- **Case.** Labels are in sentence case. Uppercase is used only for table
  column headers (11px, 0.04em tracking).
- **Alerts are the only tinted panels**, and only for real warnings or
  limitations.
- **The chat launcher is a labelled button** ("AI Assistant"), not a
  floating icon bubble over content.

## Rejected (AI-default tells)

- radial gauges
- progress bars in several colours for unrelated scores
- coloured-border KPI tiles
- card-in-card grids
- emoji icons
- gradient text
- uppercase company names
- eyebrow labels on every section
- cyan or indigo glow accents

## Verification

Before shipping a visual change:

1. Screenshot every page at 1440px and 390px, and look for overflow and
   collisions.
2. Run `npm test` and `npm run build`.
3. Run the browser verifiers in `backend/research/verify_*_ui.py`.
