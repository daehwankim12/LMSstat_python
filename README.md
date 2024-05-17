# LMSstat_python: Python implementation of LMSstat (https://github.com/daehwankim12/LMSstat)

## Installation

```bash
pip install git+https://github.com/daehwankim12/LMSstat_python.git@develop
```

## Usage

### t-test, u-test, ANOVA, and Kruskal-Wallis test

```python
from lmsstat import stats

filedir = 'your csv file directory'
result = stats.allstats(filedir)  # p-value adjusted by fdr method

result.to_csv('result.csv')
```

### Normality test

보정되지 않은 결과이므로 주의.

```python
from lmsstat import stats
import pandas as pd

path = "data.csv"
data = pd.read_csv(path)

result = stats.norm_test(data)
result
```

### Data Standardization

```python
import pandas as pd
from lmsstat import stats

path = "data.csv"
data = pd.read_csv(path)
scaled_data = stats.scaling(data)
scaled_data.to_csv("scaled_data.csv")

scaled_data
```

### PCA

```python
from lmsstat import stats, plot
import pandas as pd
from scipy.spatial import ConvexHull

data = pd.read_csv("data.csv")

pc_scores, pc_loadings, r_square = stats.pca(data)
pca_plt = plot.plot_pca(pc_scores, data['Group'])

pca_plt.display()
```

### Box plot, Dot plot

각각 현재 작업 디렉토리 밑에 만들어진 boxplots, dotplots 폴더에 자동으로 저장됨.

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plot.plot_box(data)
plot.plot_dot(data)
```

### Heatmap

클러스터링은 진행되지 않음.

```python
from lmsstat import plot
import pandas as pd

data = pd.read_csv("data.csv")

plot.plot_heatmap(data)
```