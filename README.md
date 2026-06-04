# LMSstat_python: Python implementation of LMSstat (<https://github.com/SNUFML/LMSstat>)

## Installation

```bash
pip install lmsstat
```

## Usage

**Input format**: the first two columns must be `Sample` (col 0) and `Group` (col 1). The remaining columns are treated as metabolites/features.

### t-test, u-test, ANOVA, and Kruskal-Wallis test

```python
from lmsstat import stat
import pandas as pd

data = pd.read_csv("data.csv")
result = stat.allstats(data)
# result = stat.allstats(data, p_adj=False) # When you don't want to adjust p-value

# For 3+ groups you can choose the ANOVA variance assumption and the parametric
# post-hoc test explicitly (the non-parametric Dunn post-hoc is always included):
# result = stat.allstats(data, anova_use_var="unequal")           # Welch's ANOVA
# result = stat.allstats(data, posthoc="games_howell")            # Games-Howell instead of Scheffé
# result = stat.allstats(data, posthoc="both")                    # both Scheffé and Games-Howell

result.to_csv('result.csv', index=False)  # Save the result as a csv file
```

### Normality test

보정되지 않은 결과이므로 주의.

```python
from lmsstat import stat
import pandas as pd

path = "data.csv"
data = pd.read_csv(path)

result = stat.norm_test(data)
result
```

### Data Standardization

```python
import pandas as pd
from lmsstat import stat

path = "data.csv"
data = pd.read_csv(path)
scaled_data = stat.scaling(data)
scaled_data.to_csv("scaled_data.csv")

scaled_data
```

### Preprocessing (imputation, log transform, normalization)

All preprocessing functions return a new DataFrame (the input is never mutated)
and keep the `Sample`/`Group` columns and the original feature order.

```python
import pandas as pd
from lmsstat import stat

data = pd.read_csv("data.csv")

# Missing-value imputation: "half_min" (LC-MS below-LOD), "min", or "knn"
data = stat.impute_missing(data, method="half_min")

# Sample-wise normalization: "median", "total_area", or "pqn"
data = stat.normalize(data, method="pqn")

# Log transform: log_base(x + offset); base may be 2, 10, np.e, ...
data = stat.log_transform(data, base=2, offset=1.0)
```

A typical order is impute → normalize → log transform → `scaling`, after which
`allstats`, volcano/PLS-DA, and the heatmap operate on cleaner inputs.

### PCA

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

pca_plt = plot.plot_pca(data, n_components=2)
pca_plt[0].show()
print(f"R2: {pca_plt[1]}, Q2: {pca_plt[2]}")  # R2, Q2
```

### PLS-DA

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plsda_plt = plot.plot_plsda(data, n_components=2)
plsda_plt[0].show()
print(f"R2X: {plsda_plt[1]}, R2Y: {plsda_plt[2]}, Q2: {plsda_plt[3]}")  # R2, Q2
print(plsda_plt[4].head())  # VIP ranking (DataFrame)
```

### Box plot, Bar plot
각각 현재 작업 디렉토리 밑에 만들어진 boxplot, barplot 폴더에 자동으로 저장됨.

```python
from lmsstat import plot, stat
import pandas as pd

data = pd.read_csv("data.csv")

stats_res = stat.allstats(data)

plot.plot_box(data, stats_res, test_type="t-test")
plot.plot_bar(data, stats_res, test_type="t-test")

# test_type selects which p-value column drives the significance brackets:
# "t-test", "u-test", "scheffe", "games_howell" (3+ groups), or "dunn".
# Use the same post-hoc you requested in allstats(); e.g. for games_howell:
# stats_res = stat.allstats(data, posthoc="games_howell")
# plot.plot_box(data, stats_res, test_type="games_howell")

# plot only significant metabolites for the selected test (default alpha=0.05)
# plot.plot_box(data, stats_res, test_type="t-test", significant_only=True)
# plot.plot_bar(data, stats_res, test_type="t-test", significant_only=True, alpha=0.01)

# limit parallelism (use 1 to run serially)
# plot.plot_box(data, stats_res, test_type="t-test", max_workers=1)
# plot.plot_bar(data, stats_res, test_type="t-test", max_workers=2)

# if your environment shows growing memory usage with long runs, you can restart workers periodically
# plot.plot_box(data, stats_res, test_type="t-test", max_workers=4, restart_every=500)

# boxplot point rendering options
# plot.plot_box(data, stats_res, test_type="t-test", points="none")
# plot.plot_box(data, stats_res, test_type="t-test", points="non_outliers")
```

### Heatmap

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plot.plot_heatmap(data)
```
