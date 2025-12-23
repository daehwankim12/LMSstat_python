# LMSstat_python: Python implementation of LMSstat (<https://github.com/SNUFML/LMSstat>)

## Installation

```bash
pip install lmsstat
```

## Usage

### t-test, u-test, ANOVA, and Kruskal-Wallis test

```python
from lmsstat import stat
import pandas as pd

data = pd.read_csv("data.csv")
result = stat.allstats(data)
# result = stat.allstats(data, p_adj=False) # When you don't want to adjust p-value

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

### PCA

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

pca_plt = plot.plot_pca(data)
pca_plt[0].show()
print(f"R2: {pca_plt[1]}, Q2: {pca_plt[2]}")  # R2, Q2
```

### PLS-DA

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plsda_plt = plot.plot_plsda(data)
plsda_plt[0].show()
print(f"R2X: {plsda_plt[1]}, R2Y: {plsda_plt[2]}, Q2: {plsda_plt[3]}")  # R2, Q2
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
```

### Heatmap

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plot.plot_heatmap(data)
```
