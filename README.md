# TeamKorea — MLSys 2026 Scheduling Contest

TeamKorea's agent-reasoning solution for the MLSys 2026 scheduling
contest.

## Build

Pure Python. Install the single dependency:

```bash
pip install -r requirements.txt
```

## Run

```bash
export GOOGLE_API_KEY=...
python3 agent.py <input.json> <output.json>
```

Optional environment variable: `TEAMKOREA_BUDGET_S` (default 540 s)
sets the wall-clock budget the agent respects; the default leaves a
60-second margin inside the organiser's 600-second per-benchmark
timeout.

## Tests

To smoke-test the agent on one of the released benchmarks, point it
at any `mlsys-2026-N.json` and inspect the resulting `output.json`:

```bash
GOOGLE_API_KEY=... python3 agent.py benchmarks/mlsys-2026-9.json out.json
```

The agent always writes a valid 1-op-per-subgraph fallback solution
within the first second, then iteratively improves it. Killing the
process at any point still leaves a valid `output.json` on disk.

## Benchmark Costs

Agent results on all 24 benchmarks.

| Benchmark | # Nodes | # Edges | Agent |
|-----------|--------:|--------:|---------:|
| mlsys-2026-1  |   5 |   9 |        367,002 |
| mlsys-2026-2  |   5 |   7 |         39,000 |
| mlsys-2026-3  |   4 |   6 |         72,768 |
| mlsys-2026-4  |   5 |  10 |         25,395 |
| mlsys-2026-5  |  19 |  34 |        508,644 |
| mlsys-2026-6  |  17 |  29 |        174,763 |
| mlsys-2026-7  |  15 |  21 |        129,596 |
| mlsys-2026-8  |  20 |  37 |         92,636 |
| mlsys-2026-9  |  32 |  56 |     20,731,121 |
| mlsys-2026-10 |  28 |  47 |      8,156,915 |
| mlsys-2026-11 |  26 |  38 |      1,153,683 |
| mlsys-2026-12 |  31 |  46 |      3,081,782 |
| mlsys-2026-13 |  63 | 126 |     22,156,739 |
| mlsys-2026-14 |  63 |  96 |      2,755,886 |
| mlsys-2026-15 |  61 |  97 |      1,431,241 |
| mlsys-2026-16 |  63 |  85 |      4,623,211 |
| mlsys-2026-17 | 103 | 198 |      5,025,300 |
| mlsys-2026-18 |  96 | 176 |      1,417,600 |
| mlsys-2026-19 | 103 | 154 |      1,994,769 |
| mlsys-2026-20 | 103 | 178 |      5,689,540 |
| mlsys-2026-21 | 152 | 280 |      2,158,100 |
| mlsys-2026-22 | 150 | 240 |      3,159,381 |
| mlsys-2026-23 | 121 | 186 |      3,445,107 |
| mlsys-2026-24 | 112 | 192 |      2,640,251 |

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Entry point. |
| `evaluator.py` | Local evaluator used as a scoring tool by grain search. |
| `prompts/system.md` | System prompt. |
| `prompts/repair.md` | Error-feedback template. |
| `requirements.txt` | `google-genai` only. |
| `main.tex`, `contents.tex` | Writeup source. |
| `writeup.pdf` | Compiled writeup. |

## License

Apache-2.0
