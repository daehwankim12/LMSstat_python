# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-05

**Metabolomics workflow expansion.** This release builds out an end-to-end
metabolomics pipeline — QC/RSD filtering, preprocessing, effect-size analysis,
and new visualizations — on top of a more consistent statistical engine. All
additions preserve existing defaults and APIs.

### Statistics consistency
- `allstats` now exposes explicit, reproducible options for 3+ groups:
  - `posthoc="scheffe" | "games_howell" | "both"` — the previously orphaned
    Games-Howell post-hoc is wired in; the non-parametric Dunn post-hoc is
    always included.
  - `anova_use_var="equal" | "unequal"` — Welch's ANOVA.
  - Both options are validated up front, so invalid values raise even for
    two-group input. Per-feature auto-selection is intentionally avoided so
    result columns stay reproducible.
- `anova_test` gains a `use_var` parameter.
- `plot_box` / `plot_bar` accept `test_type="games_howell"`.
- Removed the unused `pingouin` dependency (Welch's ANOVA is covered by
  `statsmodels`).

### Preprocessing & QC
- `stat.impute_missing(method="half_min" | "min" | "knn")` — missing-value
  imputation (half-min targets raw, non-negative LC-MS below-LOD values).
- `stat.normalize(method="median" | "total_area" | "pqn")` — sample-wise
  dilution normalization (PQN = integral normalization then median-quotient
  correction).
- `stat.log_transform(base, offset)` — `log_base(x + offset)`.
- `stat.rsd_filter(qc_label, max_rsd, return_rsd)` — drop features whose %RSD
  across the QC samples exceeds a threshold (the standard LC-MS reliability
  filter); feature filtering only.
- Shared contract for all of the above: the input is never mutated, and the
  `Sample`/`Group` columns, original feature order, and row index are preserved.

### Effect size & volcano
- `stat.effect_size_table(...)` — two-group per-feature table: group means,
  fold change, log2FC, Cohen's d (pooled SD), the chosen test's p-value, and a
  BH-adjusted p-value.
- `plot.plot_volcano(...)` — volcano plot that consumes the effect-size table
  (computes no statistics itself).

### Correlation & VIP plots
- `plot.plot_correlation(...)` — clustered correlation heatmap with a fixed
  `[-1, 1]` color scale for cross-run comparability.
- `plot.plot_vip(...)` — VIP bar chart consuming the PLS-DA VIP table.

### Housekeeping & testing
- `.gitignore` now covers the default plot output artifacts.
- Test suite expanded from ~150 to 281 tests.
- README documents the recommended end-to-end workflow.

### Breaking changes
- None. Existing `allstats(data)`, PCA/PLS-DA, and plotting calls behave as
  before; only the `pingouin` dependency was dropped.

[0.2.0]: https://github.com/SNUFML/LMSstat_python/releases/tag/v0.2.0
