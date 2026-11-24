# BrowseComp: sources, credits and licences

- **BrowseComp** (Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford,
  Hyung Won Chung, Alex Tachard Passos, William Fedus, Mia Glaese; OpenAI, 2025,
  https://openai.com/index/browsecomp/). The test set is downloaded at run time from
  `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv` and checked
  against its SHA-256 (`7b24471c...`); it is never vendored, and neither are decrypted questions or
  answers. Do not publish decrypted questions, answers, generated task directories or trial logs
  that quote them.
- **OpenAI simple-evals** (MIT), `browsecomp_eval.py` at
  `652c89d0ca9df547706735883097e9537d40dc47`: `QUERY_TEMPLATE`, `GRADER_TEMPLATE`, `derive_key`,
  `decrypt`, the grader settings and the `--examples N` sample are ported into
  `browsecomp_task.py`, with the `group(1)` fix of openai/simple-evals#67. Its licence follows.
- **Harbor PR #2355** "Adapter: browsecomp" (harbor-framework/harbor, Apache-2.0, by Brian Guo,
  head `464e90c108bfc3d220288c8de9b3f2d80c572b02`, Osmosis-AI/harbor:browsecomp-adapter): the
  pinned protocol module (`official.py`), the dataset checks, task-id parsing, the oracle response,
  the `browsecomp-audit` scan (extended here) and its tests, ported and adapted to this harness.
- The audit's extra indicators follow Anthropic, "Eval awareness in Claude Opus 4.6's BrowseComp
  performance" (https://anthropic.com/engineering/eval-awareness-browsecomp).

## OpenAI simple-evals licence

```
MIT License

Copyright (c) 2024 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
