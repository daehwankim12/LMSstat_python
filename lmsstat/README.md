# LMSstat_python: Python implementation of LMSstat (https://github.com/daehwankim12/LMSstat)

## Installation

```bash
pip install git+https://github.com/daehwankim12/LMSstat_python.git
```

## Usage

```python
from lmsstat import stats

filedir = 'your csv file directory'
result = stats.allstats(filedir) # p-value adjusted by Bonferroni method

result.to_csv('result.csv')
```