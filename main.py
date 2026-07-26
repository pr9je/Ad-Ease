# Import Core Libraries.
import re
import warnings
import numpy as np
import pandas as pd

# Visualixation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Stats / time series.

from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
try:
  from prophet import Prophet
except:
  !pip install prophet - q
  from prophet import Prophet

# Metrices

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.titleweight"] = "bold"


train = pd.read_csv('/content/train_1.csv')
campaign_eng = pd.read_csv('/content/Exog_Campaign_eng.csv')

print("train_1.csv")
print("shape:", train.shape)
display(train.head(3))
print(train.dtypes.value_counts())


print("\nExog_Campaign_eng.csv")
print("shape:", campaign_eng.shape)
display(campaign_eng.head(3))


PAGE_PATTERN = re.compile(
    r"(?P<name>.+)_(?P<lang>[a-z0-9\-]+)\.(?:wikipedia|wikimedia|mediawiki)\.org_"
    r"(?P<access>[a-z\-]+)_(?P<origin>[a-z\-]+)$"
)

def parse_page_regex(page: str) -> dict:
    """Extract (name, lang, access, origin) from a Wikipedia page string via regex.

    Note: the domain is not always '<lang>.wikipedia.org' -- Wikimedia Commons pages
    use 'commons.wikimedia.org' and Mediawiki help pages use 'www.mediawiki.org'. Both
    are kept (as pseudo-language buckets 'commons'/'www') rather than dropped, and the
    access/origin tokens themselves contain hyphens (e.g. 'all-access', 'all-agents'),
    so both character classes must allow '-'.
    """
    m = PAGE_PATTERN.match(page)
    if m:
        return m.groupdict()
    return {"name": np.nan, "lang": "unknown", "access": np.nan, "origin": np.nan}


def parse_page_string(page: str) -> dict:
    """Extract (name, lang, access, origin) via plain string operations."""
    try:
        left, access, origin = page.rsplit("_", 2)
        name, domain = left.rsplit("_", 1)
        lang = domain.split(".")[0]
        return {"name": name, "lang": lang, "access": access, "origin": origin}
    except ValueError:
        return {"name": np.nan, "lang": "unknown", "access": np.nan, "origin": np.nan}


sample = train["Page"].head(1000)
regex_df = pd.DataFrame(sample.apply(parse_page_regex).tolist())
string_df = pd.DataFrame(sample.apply(parse_page_string).tolist())

agreement = (regex_df["lang"] == string_df["lang"]).mean()
print(f"Regex vs string-split agreement on language field: {agreement:.2%}")

# Regex is more robust to edge cases (pages containing extra underscores); use it as
# the production approach.
# Idempotent: drop any previously-added meta columns first, so re-running this cell
# (e.g. after a Jupyter/Colab restart-less re-run) never creates duplicate columns.
train = train.drop(columns=["name", "lang", "access", "origin"], errors="ignore")
meta = train["Page"].apply(parse_page_regex).apply(pd.Series)
train = pd.concat([train, meta], axis=1)
display(train[["Page", "name", "lang", "access", "origin"]].head())

# 'commons' (Wikimedia Commons media files) and 'www' (Mediawiki help pages) are not
# language markets -- they're shared infrastructure/media domains. We keep them in the
# raw table (for completeness) but exclude them wherever we analyze/forecast "language"
# traffic, since Ad Ease's clients plan campaigns around language/region, not file pages.
NON_LANGUAGE_BUCKETS = {"commons", "www"}
print("Unique 'lang' buckets found:", sorted(train["lang"].unique()))
print("Non-language buckets excluded from language-level analysis:",
      sorted(NON_LANGUAGE_BUCKETS & set(train["lang"].unique())))