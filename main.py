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


train = train.drop(columns=["name", "lang", "access", "origin"], errors="ignore")
meta = train["Page"].apply(parse_page_regex).apply(pd.Series)
train = pd.concat([train, meta], axis=1)
display(train[["Page", "name", "lang", "access", "origin"]].head())



NON_LANGUAGE_BUCKETS = {"commons", "www"}
print("Unique 'lang' buckets found:", sorted(train["lang"].unique()))
print("Non-language buckets excluded from language-level analysis:",
      sorted(NON_LANGUAGE_BUCKETS & set(train["lang"].unique())))


# Exploratory Data Analysis (EDA)

### Missing Values
date_cols = [c for c in train.columns if re.match(r"\d{4}-\d{2}-\d{2}", str(c))]

missing_count = train[date_cols].isna().sum()
missing_pct = (missing_count / len(train) * 100).round(2)

print("Total missing cells: {:,} ({:.2f}% of all values)".format(
    missing_count.sum(), missing_count.sum() / (len(train) * len(date_cols)) * 100
))

fig, ax = plt.subplots(figsize=(14, 4))
missing_pct.plot(ax=ax)
ax.set_title("Missing Values (%) per Date")
ax.set_xlabel("Date")
ax.set_ylabel("% Missing")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

### Missing Values imputation 
def compare_imputations(series: pd.Series) -> pd.DataFrame:
    """Compare common imputation strategies on one page's time series."""
    return pd.DataFrame({
        "original": series,
        "ffill": series.ffill(),
        "bfill": series.bfill(),
        "mean": series.fillna(series.mean()),
        "median": series.fillna(series.median()),
        "linear_interp": series.interpolate(method="linear"),
        "rolling_avg_7": series.fillna(series.rolling(7, min_periods=1, center=True).mean()),
    })

example_page = train.loc[train[date_cols].isna().sum(axis=1) > 0, date_cols].iloc[0]
example_page.index = pd.to_datetime(example_page.index)
comparison = compare_imputations(example_page)

comparison[["original", "ffill", "linear_interp", "rolling_avg_7"]].plot(figsize=(14, 5))
plt.title("Imputation Strategy Comparison — Example Page")
plt.xlabel("Date"); plt.ylabel("Views")
plt.legend()
plt.tight_layout()
plt.show()


train[date_cols] = train[date_cols].interpolate(method="linear", axis=1, limit_direction="both")


### Duplicate Records
n_dupes = train.duplicated(subset=['Page']).sum()
print(f"Duplicate Page rows: {n_dupes}")
train = train.drop_duplicates(subset=['Page']).reset_index(drop=True)

### Language Distribution
lang_counts = train.loc[~train["lang"].isin(NON_LANGUAGE_BUCKETS), "lang"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
lang_counts.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Number of Pages per Language")
axes[0].set_xlabel("Language"); axes[0].set_ylabel("Page Count")

axes[1].pie(lang_counts, labels=lang_counts.index, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Language Share of Pages")
plt.tight_layout
plt.show()

# Access Type & Access Origin Distribution

