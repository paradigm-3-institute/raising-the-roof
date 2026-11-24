# Chart style

Every animated chart in the post uses the kit in `_kit/` (kit.css, kit.js, Instrument Sans under the SIL Open Font
License in `_kit/fonts/`), so the charts look like one set. A chart page loads `../_kit/kit.css` and `../_kit/kit.js`,
then defines its DATA, text, beats and `draw(t)`, and ends with `boot()`. Opening `chart.html?t=<seconds>` shows the
frame at that time; the GIFs are recordings at 1080 px wide.

## Colors

Each color has one meaning in every chart. A setup's color never means anything else. No other hue appears in a chart. There is no red "punch" color and no green "good" color.

| Setup | fill | line | text | tint |
| --- | --- | --- | --- | --- |
| Single agent (one agent, one session) | `#8a867c` | `#63615a` | `#585650` | `#b9b6b0` |
| Restarted single agent (one agent, budget split into sessions run one after another, each with a fresh context) | `#2f8f7f` | `#2a7266` | `#28665c` | `#82bcb2` |
| Protocol teams | `#8fa3be` | `#667385` | `#5b6674` | `#bcc8d8` |
| Self-organized teams | `#d4765a` | `#a65f4a` | `#935644` | `#e5ad9c` |
| Orchestrated teams | `#cdb894` | `#7d715d` | `#6b6251` | `#e1d4bf` |
| No-talk teams (agents working independently, scored on the best one) | `#6b5b95` | `#5b4e7c` | `#534870` | `#a69dbf` |

- **fill**: bars, run dots, areas.
- **line**: strokes up to 4 px, mean dots, intervals, reference lines, thick lines in line charts.
- **text**: the setup's name and values. Contrast on white is 5.75:1 or more.
- **tint**: a variant of the same setup, such as another Codex build or another budget. It is always labeled.

The line and text shades mix the fill with ink. In code, use `SETUP.selforg.fill`, `.line`, `.text`, `.tint`, `.name`.

The no-talk violet is new. No chart used one before.

**Mixed Claude and GPT teams** have no color of their own. A mixed team is still a protocol or self-organized team, so it takes that setup's color. It is marked with a diamond.

**Models are never shown by color.** Use these, in this order:

1. Position: one panel or one row group per model, headed by the model's full name in ink.
2. Labels: the model's full name in ink or muted text next to its mark.
3. Shape, only when two models share one row or one plot: circle for GPT-6.1 Sol and GPT-6.1 Terra, square for Claude Opus 5.5 and Claude Sonnet 5.5, diamond for a mixed team. The shape takes the setup's color. `MODEL.<key>.shape` holds the shapes, and `shape()`, `run()` and `mean()` draw them.

Neutrals:

| Token | Hex | Use |
| --- | --- | --- |
| `ink` | `#1b1b1a` | title, row names before their beat, labels inside bars, callouts |
| `muted` | `#6f6d66` | row notes, ticks, axis titles, footnote |
| `faint` | `#a9a69e` | de-emphasized labels |
| `grid` | `#ecebe6` | gridlines, empty tracks |
| `axis` | `#b9b6ae` | baselines, leader lines, a reference that is not a setup |
| `context` | `#bcb8ae` | bands and ranges that are not a setup, such as an outside estimate |
| `bg` | `#ffffff` | page |

### Color checks

All pairs of the six fills, on white, checked with a palette validator. Distances are OKLab ΔE ×100. The target is 8 or more under color blindness and 15 or more with normal color vision.

- Single agent and Restarted single agent: protan **0.4**, deutan 4.1, normal 9.8. A viewer with protanopia sees the same color.
- Single agent and Self-organized teams: protan 4.8, normal 12.4.
- Restarted single agent and Self-organized teams: protan 5.2.
- Single agent and Protocol teams: normal 10.8. Protocol and Orchestrated teams: normal 12.9.
- Every pair with No-talk teams is 11 or more under color blindness and 15 or more with normal vision.
- On white, the Protocol fill is 2.58:1 and the Orchestrated fill is 1.93:1. Both are fine for bars and dots. Thin strokes therefore use the line shade, which is 4.78:1 or more.

The validator also fails the palette on chroma and the lightness band. That is expected: it was made for saturated palettes, and this one is muted on purpose. Because of these results, **color is never the only cue**. Every setup's marks carry its name, either next to the mark or in the row label. Single agent and Restarted single agent are always told apart by position or label too.

## Canvas, type and spacing

- 1080×1350, 30 fps, white page, Instrument Sans with tabular figures. kit.css loads the font from `_kit/fonts/`.
- Frame padding: 54 top, 44 right, 40 bottom, 60 left. 30 px from header to chart, 26 px from chart to footnote. Nothing within 16 px of an edge.
- Type, in px (`TYPE` in kit.js):

| Role | Size | Weight | Color |
| --- | --- | --- | --- |
| Title | 31 | 600 | ink, at most 2 lines |
| Row or panel name | 27 | 700 | ink at frame 0, then the setup's text shade when its beat starts; ink for a model |
| Row note, label inside a bar, row label in a dense chart | 23 | 400 / 600 | muted / ink |
| Label beside a mark, callout words, legend | 22 | 600 / 400 | setup text shade, or ink |
| Small label on a mark | 21 | 600 | setup text shade |
| Ticks, axis titles | 20 | 400 | muted |
| Data value | 24 | 700 | setup text shade |
| Hero number (the callout) | 31 | 700 | ink |
| Footnote | 18.5, line height 1.47 | 400 | muted |

Nothing a viewer must read is under 18 px.

## Text

- **The title carries the finding.** It is one sentence in sentence case with a verb and a number, no final period, and at most 2 lines.
- **No subtitle**, unless the chart needs one to name the measure and its unit. Then it is one line, and it never repeats the title.
- **No legend row** if direct labels can do the job. Name setups in row labels or next to their marks, in their text shade. A legend is only for setups or shapes that repeat across many marks. Then it is one row under the title, built from `TEXT.key`.
- **Footnote: at most 2 short lines.** Source, runs per arm, method, and anything left out.
- **No em dashes.** Write ranges as "4 to 10", or "4–10" with an en dash and no spaces.
- **Names, exactly:** Single agent, Restarted single agent, Protocol teams, Self-organized teams, Orchestrated teams, No-talk teams. Models: GPT-6.1 Sol, GPT-6.1 Terra, Claude Opus 5.5, Claude Sonnet 5.5. Never "Sol" or "Opus 5.5" alone. Tasks: git-to-zig, AHC014, AHC038, Performance take-home.
- **Code keys are not names:** `ttc`, `board`, `peers`, `chat`, `ultra`, `relay` and `notalk` appear only in code and data; a chart uses the names above. The message board can be named as a channel.

## Marks

`MARK` in kit.js holds these values, and the helpers draw them.

- **Run**: a dot of radius 7 in the setup's fill, with a 1.5 px white ring. `run()`.
- **Mean of runs**: a dot of radius 11 in the line shade, with a 3 px white ring, drawn on top. Use radius 8 when rows are under 32 px apart. `mean()`.
- **Run that did not spend its budget** (Claude agents that stopped early): the same size, white inside, with a 2.5 px ring in the line shade. Say what it means in the footnote or legend. `run(..., { hollow: true })`.
- **95% interval**: a 4 px line in the line shade, with round ends, at the mean's height. It grows out from the mean and is drawn over the run dots. `interval()`.
- **Line charts**: the mean line is 3.5 px in the line shade. Each run is a 1.5 px line in the fill, at 35% opacity.
- **Bars**: fill, 3 px corner radius, 4 px gaps between parts.
- **Gridlines**: 1.5 px in grid, along the value axis only. No tick marks and no value-axis line. A 2 px axis-gray baseline at zero on bar and share axes. Tick labels are 20 px muted, 14 px left of the plot or 28 px below it. The axis title sits at the far end of its axis. `grid()`.
- **Reference line**: a level the chart compares against.
  - If it stands for a setup, such as the single agent's mean or 100% of the best single agent, it is 2 px, dashed `6 5`, in that setup's line shade. It is labeled with the setup's name at its top end, 22 px 600, in the text shade.
  - If it is not a setup, such as 0 or λ = 1, it is 2 px solid in axis gray, labeled in muted. `refLine()`, `tag()`.
- **Off-scale value**: a short arrow at the plot edge with the value ("> 1").

## Callout

There is one style for gaps and highlighted comparisons:

- A dotted connector in ink, 2.5 px, dash `1 6`, round caps. It is a straight line, an elbow or a bracket between the two levels being compared, and it is drawn under the marks.
- The number goes beside the gap: 31 px 700 in ink, followed by at most 6 words in 22 px muted on the same line ("+9.9 pts vs team", "8.5x sooner, 9.9 pts lower"). It has a white halo, and no line runs through it.
- The number is computed in code. Each chart measures against one reference, and each panel has one callout.
- `callout(parent, points, u, { hero, words, x, y, anchor })`.

## Motion

`MOTION` in kit.js holds the values.

- **Runtime 16 s.** Use 15 s if the build ends by 7.5 s.
- **The build ends by 8.5 s**, so the final hold is at least 45% of the runtime (the reference ends at 8.1 s, a 49% hold). Nothing moves during the hold.
- **Frame 0** shows the title, the footnote, row names in ink, axes, gridlines and empty tracks.
- **Beats**: the first starts at 0.4 s. Each lasts 1.2 to 2.1 s, with 0.3 to 0.5 s between beats. The order is the context (usually the single agent), then the other setups top to bottom, then the subject, then the callout.
- **Easing**:
  - Bars sweep in with `ease.inOut` over 0.6 to 0.9 s.
  - Dots and nodes pop with `ease.out` over 0.25 s.
  - Labels fade in over 0.25 to 0.3 s, once their mark is about 70% grown.
  - A row name turns from ink to its setup color over 0.5 s as its beat starts, and its note fades in over 0.4 s.
  - A callout's line draws first, then its number fades in.
- No bounce, no camera moves, no title fade-in.
