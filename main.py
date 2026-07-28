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

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
train["access"].value_counts().plot(kind="bar", ax=axes[0], color="darkorange")
axes[0].set_title("Access Type Distribution (Desktop / mobile-web / all-access)")
axes[0].set_xlabel("Access Type"); axes[0].set_ylabel("Page Count")

train["origin"].value_counts().plot(kind="bar", ax=axes[1], color="seagreen")
axes[1].set_title("Access Origin Distribution (spider vs browser agent)")
axes[1].set_xlabel("Access Origin"); axes[1].set_ylabel("Page Count")
plt.tight_layout()
plt.show()

# Daily Total Views (All pages, all languages)
daily_total = train[date_cols].sum(axis=0)
daily_total.index = pd.to_datetime(daily_total.index)

plt.figure(figsize=(14, 5))
daily_total.plot()
plt.title("Total Wikipedia Page Views - All Pages, All Languages")
plt.xlabel("Date"); plt.ylabel("Total Views")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

growth = (daily_total.iloc[-30:].mean() / daily_total.iloc[:30].mean() -1) *100
volatility = daily_total.pct_change().std() * 100
print(f"Growth (last 30d avg vs first 30d avg): {growth:.1f}%")
print(f"Day-over-day volatility (std of pct change): {volatility:.2f}%")


# Average Views by Language
avg_by_lang = (train.loc[~train["lang"].isin(NON_LANGUAGE_BUCKETS)].groupby("lang")[date_cols].sum().sum(axis=1).sort_values(ascending=False))

plt.figure(figsize=(12, 5))
avg_by_lang.plot(kind='bar', color='teal')
plt.title("Total Views by Language (entire period)")
plt.xlabel("Language"); plt.ylabel("Total Views")
plt.tight_layout()
plt.show()

print("Top 3 languages by traffic:\n", avg_by_lang.head(3))
print("\n Lowest 3 languages by traffic:\n", avg_by_lang.tail(3))

# Outliers Detection

lang_daily = (train.loc[~train["lang"].isin(NON_LANGUAGE_BUCKETS)].groupby("lang")[date_cols].sum().T)
lang_daily.index = pd.to_datetime(lang_daily.index)

fix, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(data=np.log1p(lang_daily), ax=axes[0])
axes[0].set_title("Log(1+Views) Distribution by Language (Boxplot)")
axes[0].tick_params(axis="x", rotation=45)

top_lang = avg_by_lang.index[0]
sns.histplot(lang_daily[top_lang], kde=True, ax=axes[1], color="purple")
axes[1].set_title(f"Distribution of Daily Views — '{top_lang}'")
plt.tight_layout()
plt.show()

def iqr_outlier_bounds(series: pd.Series, k:float = 1.5) -> tuple:
  q1, q3 = series.quantile([0.25, 0.75])
  iqr = q3 - q1
  return q1 - k * iqr, q3 + k * iqr

low, high = iqr_outlier_bounds(lang_daily[top_lang])
n_outliers = ((lang_daily[top_lang] < low) | (lang_daily[top_lang] > high)).sum()
print(f"IQR outlier bounds for '{top_lang}':[{low:.0f}, {high:.0f}] -> {n_outliers} outlier days")

# Data Transformation

# Demonstrate melt on a manageable sample (illustrates wide -> long technique)
sample_for_melt = train.sample(500, random_state=RANDOM_STATE)
long_sample = sample_for_melt.melt(
    id_vars=["Page", "name", "lang", "access", "origin"],
    value_vars=date_cols,
    var_name="date",
    value_name="views",
)
long_sample["date"] = pd.to_datetime(long_sample["date"])
print("Long-format sample shape:", long_sample.shape)
display(long_sample.head())

# Efficient full-scale aggregation to the language grain (equivalent to melt+pivot, but avoids materializing an ~80M-row long dataframe)
lang_pivot = (train.loc[~train["lang"].isin(NON_LANGUAGE_BUCKETS)]
              .groupby("lang")[date_cols].sum().T)
lang_pivot.index = pd.to_datetime(lang_pivot.index)
lang_pivot = lang_pivot.sort_index().asfreq("D")  # enforce daily frequency, exposes gaps
display(lang_pivot.head())