# CLAP-dependent tables from the seeded run (2026-11-07)

Every table in `data/` that depends on CLAP (`clap`, `clap_n`), recomputed from the seeded CLAP (`clap`, `clap_n`) in
`data/clap_seeded_2026-11-07.csv`. They come from the same Kaggle run (`rerun/step2/kaggle_out/`), with
all other columns unchanged. These are **not** banked: `data/` keeps the original CLAP until a decision is
made on which run the paper cites.

| file | contents |
|---|---|
| `s6_bootstrap_cis_clap_rows.csv` | `droop_L9-L7_clap` per model (point, 95% CI), seeded |
| `s6_level_means_clap.csv` | per model × phrasing × level means of `clap`, `clap_n`, seeded |
| `../clap_droop_old_vs_new.csv` | droop CIs old vs seeded, with verdicts (0 verdict changes) |
| `../clap_cell_means.csv` | cell means old vs seeded |
| `../clap_shift_by_model.csv` | per-model mean shift, max clip and cell-mean change, Spearman old↔new |
