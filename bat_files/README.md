# bat_files

Windows launchers for the pipeline, one per stage, in run order. Each activates the conda env
itself, so any of them can be double-clicked or run from a bare `cmd.exe` with no setup beyond
`setup_env.bat`.

| Script | Stage | Where it runs |
|---|---|---|
| `setup_env.bat` | Create the env, install deps **and the project** | Anywhere, once |
| `01_data_pipeline.bat` | 1 — corpus: extract, clean, merge, metrics | CPU fine |
| `02_tokenizers.bat` | 2 — corpus variants, then all 12 tokenizers | CPU fine |
| `03_mlm_pretrain.bat` | 3 — MLM pre-training | **GPU only** |
| `04_export_backbone.bat` | 3b — build `mlm/<run>/final/` from a checkpoint | CPU fine |
| `05_nli_finetune.bat` | 4 — NLI fine-tuning | **GPU only** |
| `_common.bat` | Shared preamble, not run directly | — |

Stage 5 (STS) and 6 (evaluation) have no script because they have no code — they are blocked
on the absence of a Tamil–Tamil STS dataset, not on effort. See the
[README](../README.md#the-missing-piece-a-tamiltamil-sts-set).

`04_export_backbone.bat` is off the main path. `03_mlm_pretrain.bat` writes `final/` itself; the
export script is only needed for runs that predate that step, which includes
`03_sandhi_codepoint_bert`. So NLI currently needs `04` run first.

## Env vars

| Variable | Default | Use |
|---|---|---|
| `TAMBERT_ENV` | `tambert_env` | Conda env name. The CPU-only dev box uses `tambert` |
| `TAMBERT_ALLOW_CPU` | unset | Skip the CUDA guard on stages 03/05. You must remove `fp16=True` from the script first, or it will still fail |
| `TAMBERT_NO_PAUSE` | unset | Don't `pause` on failure. Set it when chaining these from another script |

```bat
set TAMBERT_ENV=tambert
bat_files\02_tokenizers.bat
```

## What each script checks before doing any work

- **Repo root.** `_common.bat` cds to `%~dp0..`, not `%~dp0`. The scripts invoke
  `python scripts/<pkg>/<file>.py`, which only resolves from the root.
- **Editable install.** `python -c "import paths"`. `pip install -e .` is mandatory and cannot
  live in `requirements.txt`; without it every stage dies on `ModuleNotFoundError`, but only
  after loading torch, so the real cause ends up buried.
- **CUDA**, on stages 03 and 05 only, because both set `fp16=True`.

## History

These were rewritten in full because all three originals were unrunnable:

- Every one did `cd /d "%~dp0"`, parking the CWD in `bat_files\`, then called
  `python scripts/...` — a path that only exists one level up. None of them could ever have
  worked.
- `setup_env.bat` then ran `pip install -r requirements.txt` from that same wrong directory,
  and never ran `pip install -e .` at all.
- `sandhi_grapheme_generator.bat` activated `tambert`, an env `setup_env.bat` does not create;
  pointed at pre-refactor module paths (`scripts/tokenizer/*_precompute.py`, now under
  `core/`); and trained only the BPE family. `02_tokenizers.bat` replaces it.

The shared preamble exists because those three each carried their own copy of the activate
block and had already drifted apart. One copy cannot drift.
