# Analysis

The scripts behind the post's numbers and charts. Run them from the repo root with Python 3.11+. They read only
[`../data/`](../data/) and need no packages.

```
python3 analysis/headline.py
```

| The post says | Script | Prints | Reads |
|---|---|---|---|
| Over 1,100 runs | [`headline.py`](headline.py) | 1,102 counted runs | `runs.csv` |
| More than 80 team configurations compared with a single agent | [`headline.py`](headline.py) | 82 (78 at 8M, 4 at 32M) | `runs.csv` |
| Teams scored below a single agent in 54 of 70 comparisons (tokens chart) | [`headline.py`](headline.py) | 54 of 70; by task 21/28, 12/12, 15/15, 6/15 | `runs.csv` |
| Teams of 8 scored about 7% lower; ~93% of the result | [`headline.py`](headline.py) | median −7.0% over 24 counted teams of 8 | `runs.csv` |
| Teams of 8 finished about 6 times sooner | [`headline.py`](headline.py) | median 6.12x over 22 teams of 8 (3.24x to 9.90x) | `runs.csv` |
| Footnote 4: 3 to 16 times sooner, except Claude Sonnet 5.5 on the take-home; Opus single agents at the 240-hour cap | [`headline.py`](headline.py) | 0.93x (Sonnet, take-home) and 3.2x to 15.5x for the rest; 3 Opus single agents at 240 h | `runs.csv` |
| Sonnet's single agents spent 31% of their budget on AHC038, 30–43% across the AHC tasks and the take-home; Opus's about three quarters on the AHC tasks | [`headline.py`](headline.py) | 43%, 31%, 30%; Opus 66% and 89% (77% together) | `runs.csv` |
| Claude Opus 5.5 ranked first on AHC014 in most runs | [`headline.py`](headline.py) | single agent 2 of 3, protocol teams 8 of 9 at 3389 | `runs.csv` |
| At 32M: Sol's teams of 8 93.2% vs 83.7%; Terra's teams of 2, 8 and 4 29.1%, 32.0%, 25.4% vs 26.6%; teams of 16 33.3%, of 32 23.2% | [`gitzig32.py`](gitzig32.py) | the same | `runs.csv` |
| Footnote 5: intervals for the gap, in points | [`gitzig32.py`](gitzig32.py) | Sol 8: 3.7 to 15.7; Terra 8: 2.3 to 8.4, 2: −0.6 to 5.6, 4: −4.4 to 1.9, 16: 0.7 to 12.5, 32: −7.3 to 0.5 | `runs.csv` |
| Patched-build reruns: Sol 89.7% vs 93.2%, Terra 31.5% vs 32.0% | [`gitzig32.py`](gitzig32.py) | the same | `runs.csv` |
| Best of 8 single agents scored after 4M: 43.7% (Sol), 13.1% (Terra) | [`gitzig32.py`](gitzig32.py) | 43.7%, 13.1% | `runs.csv`, `snapshots.csv` |
| Restarted single agent: 97.0% and 41.9% at 32M against 93.2% and 32.0%; 62.1% and 28.0% at 8M against 55.0% and 17.8%; it also beat Terra's teams of 16 and 32 | [`restart.py`](restart.py) | the same | `runs.csv` |
| λ against the restarted single agent: 0.94 (Sol) and 0.75 (Terra) at 32M, 0.89 and 0.60 at 8M; against the single agent above 1 at 32M, about 1 at 8M | [`restart.py`](restart.py), [`lam.py`](lam.py) | 0.94, 0.75, 0.89, 0.60; 1.21, 1.36, 1.04, 1.09 | `runs.csv`, `snapshots.csv` |
| Terra's restarted agent made about 60% more calls at 8M; Sol's about as many at 32M and scored 13 points higher | [`restart.py`](restart.py) | +58%; +6% and +13.3 points | `runs.csv` |
| About 3.5 times sooner for 2.3 times the tokens (Terra, 8M); about 7 times sooner for 13% more tokens (Sol, 32M) | [`restart.py`](restart.py) | 3.5x for 2.30x; 7.1x for 1.13x | `runs.csv`, `snapshots.csv` |
| GPT-6.2 Sol: teams of 2 and 4 slightly below the single agent, teams of 8 24% below | [`newer_models.py`](newer_models.py) | −2.4%, −4.3%, −23.7% | `runs.csv` |
| Cost to replicate: $70,000 at API pricing | [`cost.py`](cost.py) | $68,188 for every run here, $64,930 for the counted runs | `tokens.csv`, `prices.csv` |
| Methodology: which model ran which setup, repeats per configuration, Codex builds; the token weighting | [`methodology.py`](methodology.py) | coverage table; 3 runs for most configurations, 10 or 5 for some; weighted tokens recomputed from the token columns for 1,162 of 1,163 runs | `runs.csv` |
| Diversity appendix: Sol 53.3% vs Sonnet 41.7%; mixed teams between their Claude-only and Sol-only teams in 8 of 12; +1.9% on AHC014 (teams of 2) and the take-home (teams of 8); 16 of 30 AHC014 runs at 3389; Opus single agents spent 3.6M to 6.9M, two at the time cap | [`diversity.py`](diversity.py) | the same | `runs.csv` |
| Communication appendix: talking added 16 to 53 points for teams of 4 and 8 on git-to-zig; Sol's self-organized team of 8 +53 vs protocol +20; on the AHC tasks talking cost GPT's teams 2 to 12 points in all 18 comparisons; take-home within about 4 points; no-talk teams of 8 31% to 60% below on git-to-zig; protocol teams of 8 won back 18 to 26 points and finished 27% to 41% below | [`communication.py`](communication.py) | the same (take-home −3.5 to +3.7) | `runs.csv` |
| Newer-models appendix: Opus 5 lost 7 to 10%, Opus 5.5 5 to 8% on AHC038 | [`newer_models.py`](newer_models.py) | −7.2%, −10.0%, −7.2%; −4.7%, −6.8%, −8.2% | `runs.csv` |
| BrowseComp appendix: 67.6% and 80.2% (single agent, low and high effort), 74.3% and 83.0% (orchestrated), 80.6% (protocol), 81.0% (self-organized); gains of 0.4 to 2.8 points vs OpenAI's 2.6; 1.4 and 7.5 to 8.0 points below OpenAI; about ±5 points | [`browsecomp.py`](browsecomp.py) | the same; ±4.5 to ±5.7; with the 19 trials flagged `suspected-contamination` counted wrong, 67.2, 79.4, 73.1, 81.0, 78.3, 80.2 | `browsecomp_trials.csv` |
| BrowseComp: GPT-4.1's verdict on every final answer, behind those scores | [`browsecomp/regrade.py`](browsecomp/regrade.py) | writes `browsecomp_trials.csv`; needs the run folders, which are not public, and an OpenAI API key | run folders |

## Charts

| Chart in the post | Folder | `tally.py` prints |
|---|---|---|
| How a budget is split (Method) | [`figures/method/`](figures/method/) | no data; the page holds the budgets and token weights only |
| Performance against tokens | [`figures/tokens/`](figures/tokens/) | the curves and the 82 comparisons: 54 of 70 below, −7.0% median for teams of 8 |
| Performance against time | [`figures/time/`](figures/time/) | the same comparisons by times sooner: 6.1x and 7.0% lower for teams of 8 (3.2x to 9.9x) |
| The restarted single agent | [`figures/restart/`](figures/restart/) | the four panels' curves and finals: +3.8, +9.9, +7.1, +10.2 points vs the team of 8 |
| Mixed teams (diversity appendix) | [`figures/diversity/`](figures/diversity/) | 8 of 12 between; +1.9% twice |
| Talking vs no-talk teams (communication appendix) | [`figures/communication/`](figures/communication/) | every "talking added" value; +53 points; 9 of 9 lower on each AHC task |

The post's other two images are not in this folder: the λ trade-off diagram in Takeaways (its four points are
`restart.py`'s λ and trade-offs) and the table of team compositions in the diversity appendix (its numbers,
53.3% and 41.7%, are `diversity.py`'s).

`python3 analysis/figures/<name>/tally.py --write` pastes the numbers into that chart's page.
[`figures/STYLE.md`](figures/STYLE.md) is the charts' shared style; the kit is in [`figures/_kit/`](figures/_kit/).

## Shared code

[`common.py`](common.py) loads the data and holds the rules every script, chart and the results page share: a
configuration is one task, model, setup, team size, budget and variant; its score is the mean final of its counted
runs; a gap is team / single agent − 1 (take-home as speed), with a bootstrap 95% interval (10,000 draws, runs
resampled within each arm); a comparison counts only if the single agent spent at least 85% of its budget.
[`lam.py`](lam.py) computes λ. [`configs.py`](configs.py) builds the table of every configuration with its default
comparison (`python3 analysis/configs.py --csv configs.csv`); the results page (`python3 site/build.py`, about a
minute, writes `site/out/index.html`) is built from the same code. Each chart's folder in [`figures/`](figures/) has the self-playing page
(`chart.html`), the recording the post shows (`chart.gif`) and `tally.py`, which prints the numbers the page shows.

Intervals are seeded bootstraps, so they can differ from the post's in the last digit.
