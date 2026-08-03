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



# Time Series Visualization

# Plot a series with weekly/monthly rolling means overlaid.
def plot_series_with_rolling(series: pd.Series, title: str, windows=(7, 30)):
    plt.figure(figsize=(14, 5))
    plt.plot(series, label="Daily", alpha=0.4)
    for w in windows:
        plt.plot(series.rolling(w).mean(), label=f"{w}-day rolling mean")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Views")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

top_langs = avg_by_lang.head(4).index.tolist()

plt.figure(figsize=(14, 6))
for lang in top_langs:
    plt.plot(lang_pivot.index, lang_pivot[lang], label=lang)
plt.title("Daily Views — Top 4 Languages Overlaid")
plt.xlabel("Date"); plt.ylabel("Views"); plt.legend()
plt.tight_layout()
plt.show()

for lang in top_langs[:2]:
    plot_series_with_rolling(lang_pivot[lang], f"'{lang}' Views with 7d/30d Rolling Mean")


# Stationarity Tests

def run_stationarity_tests(series: pd.Series, name: str = "series") -> dict:
    """Run ADF + KPSS and return a summary dict."""
    series = series.dropna()
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    result = {
        "name": name,
        "adf_stat": adf_stat, "adf_p": adf_p,
        "adf_stationary": adf_p < 0.05,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,
        "kpss_stationary": kpss_p > 0.05,
    }
    print(f"[{name}] ADF p={adf_p:.4f} (stationary={result['adf_stationary']}) | "
          f"KPSS p={kpss_p:.4f} (stationary={result['kpss_stationary']})")
    return result

target_lang = top_langs[0]
series = lang_pivot[target_lang].interpolate()

_ = run_stationarity_tests(series, f"{target_lang} - raw")

d = 0
diff_series = series.copy()
while True:
    res = run_stationarity_tests(diff_series, f"{target_lang} - diff order {d}")
    if res["adf_stationary"] and res["kpss_stationary"]:
        break
    d += 1
    diff_series = series.diff(d).dropna()
    if d > 3:
        break

print(f"\nStationarity achieved at differencing order d={d}")


# Time Series Decomposition
def plot_decomposition(series: pd.Series, model: str, period: int =7, title: str=""):
  decomp = seasonal_decompose(series.dropna(), model=model, period=period)
  fig = decomp.plot()
  fig.set_size_inches(14, 8)
  fig.suptitle(f"{title} - {model.capitalize()} Decomposition", y=1.02, fontweight='bold')
  plt.tight_layout()
  plt.show()
  return decomp

decomp_add = plot_decomposition(series, "additive", period=7, title=target_lang)
decomp_mul = plot_decomposition(series, "multiplicative", period=7, title=target_lang)

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_pacf(series.dropna(), lags=30, ax=axes[0], method='ywm')
axes[0].set_title(f"PACF (raw) - {target_lang}")

plot_acf(diff_series.dropna(), lags=30, ax=axes[1])
axes[1].set_title(f"ACF (differnced, d={d}) - {target_lang}")
plt.tight_layout()
plt.show()

# Train-Test Split
TEST_DAYS = 42 # 6 full weeks - multiple of the 7-days sesonal period.

train_series = series.iloc[:-TEST_DAYS]
test_series = series.iloc[-TEST_DAYS:]

print(f"Train range: {train_series.index.min().date()} -> {train_series.index.max().date()} "
      f"({len(train_series)} days)")
print(f"Test range : {test_series.index.min().date()} -> {test_series.index.max().date()} "
      f"({len(test_series)} days)")

# Baseline Models:
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
  return np.repeat(train.iloc[-1], horizon)

def moving_average_forecast(train: pd.Series, horizon: int, window: int =7) -> np.ndarray:
  return np.repeat(train.iloc[-window:].mean(), horizon)

def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    last_cycle = train.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_cycle, reps)[:horizon]

baselines = {
    "Naive": naive_forecast(train_series, TEST_DAYS),
    "Moving Average (7d)": moving_average_forecast(train_series, TEST_DAYS),
    "Seasonal Naive (7d)": seasonal_naive_forecast(train_series,TEST_DAYS)
}

# Reusable Evaluation Helper
# Compute MAE, RMSE, MAPE, R^2, for a forecast and return a dict row
def evaluate_forecast(y_true: pd.Series, y_pred: np.ndarray, model_name:str) -> dict:
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true ))) * 100
  r2 = r2_score(y_true, y_pred)
  return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE':mape, "R2": r2 }

results = []
for name, preds in baselines.items():
  results.append(evaluate_forecast(test_series, preds, name))

pd.DataFrame(results).round(3)

# Small grid search over (p, d, q), return (best_order, best_model, results_df)

def grid_search_arima(train: pd.Series, p_range, d_range, q_range):
  rows = []
  best_aic, best_order, best_model = np.inf, None, None

  for p in p_range:
    for d_ in d_range:
      for q in q_range:
        try:
          m = ARIMA(train, order=(p, d_, q)).fit()
          rows.append({"order": (p,d_,q), "aic": m.aic})
          if m.aic < best_aic:
            best_aic, best_order, best_model = m.aic, (p, d_, q), m
        except Exception:
          continue
  return best_order, best_model, pd.DataFrame(rows).sort_values("aic")

best_order, arima_model, arima_grid = grid_search_arima(train_series, p_range=(0, 4), d_range =[d], q_range = (0,4))
print("Best ARIMA order: ", best_order, " | AIC:", arima_model.aic)

arima_forecast = arima_model.forecast(TEST_DAYS)
results.append(evaluate_forecast(test_series, arima_forecast, f"ARIMA{best_order}"))

plt.figure(figsize=(14, 5))
plt.plot(train_series.index, train_series, label='Train')
plt.plot(test_series.index, test_series, label='Actual')
plt.plot(test_series.index, arima_forecast, label=f"ARIMA{best_order} Forecast")
plt.title(f"ARIMA Forcast - {target_lang}")
plt.legend(); plt.tight_layout(); plt.show()

# SARIMA
def fit_sarima(train: pd.Series, order, seasonal_order):
  return SARIMAX(
      train, order=order, seasonal_order=seasonal_order,
      enforce_stationarity=False,
      enforce_invertibility=False,
  ).fit(disp=False)

seasonal_order = (1, 1, 1, 7) # weekly seasonality
sarima_model = fit_sarima(train_series, order=best_order, seasonal_order=seasonal_order)
sarima_forecast = sarima_model.forecast(steps=TEST_DAYS)

results.append(evaluate_forecast(test_series, sarima_forecast, f"SARIMA{best_order}x{seasonal_order}"))

plt.figure(figsize=(14, 5))
plt.plot(train_series.index, train_series, label="Train")
plt.plot(test_series.index, test_series, label="Actual")
plt.plot(test_series.index, sarima_forecast, label="SARIMA Forecast")
plt.title(f"SARIMA Forecast - {target_lang}")
plt.legend(); plt.tight_layout(); plt.show()

# SARIMAX 
english_series = lang_pivot["en"].interpolate() if "en" in lang_pivot.columns else series

# Exog_Campaign_eng has no date column of its own -- it is one flag per day, in the
# same 550-day order as the date columns in train_1.csv, so we attach dates positionally.
campaign_flags = campaign_eng.iloc[:, 0].values
campaign_dates = pd.to_datetime(date_cols)
campaign_series = pd.Series(campaign_flags, index=campaign_dates).reindex(english_series.index).fillna(0)

eng_train, eng_test = english_series.iloc[:-TEST_DAYS], english_series.iloc[-TEST_DAYS:]
camp_train, camp_test = campaign_series.iloc[:-TEST_DAYS], campaign_series.iloc[-TEST_DAYS:]

# Without campaign (SARIMA on English)
sarima_eng = fit_sarima(eng_train, order=best_order, seasonal_order=seasonal_order)
sarima_eng_fc = sarima_eng.forecast(steps=TEST_DAYS)

# With campaign (SARIMAX)
sarimax_eng = SARIMAX(
    eng_train, exog=camp_train, order=best_order, seasonal_order=seasonal_order,
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)
sarimax_eng_fc = sarimax_eng.forecast(steps=TEST_DAYS, exog=camp_test.values.reshape(-1, 1))

results.append(evaluate_forecast(eng_test, sarima_eng_fc, "SARIMA (English, no campaign)"))
results.append(evaluate_forecast(eng_test, sarimax_eng_fc, "SARIMAX (English + campaign)"))

plt.figure(figsize=(14, 5))
plt.plot(eng_test.index, eng_test, label="Actual", linewidth=2)
plt.plot(eng_test.index, sarima_eng_fc, label="SARIMA (no campaign)", linestyle="--")
plt.plot(eng_test.index, sarimax_eng_fc, label="SARIMAX (+ campaign)", linestyle="--")
plt.title("English Views — Impact of Campaign Exogenous Variable")
plt.legend(); plt.tight_layout(); plt.show()

# Forecasting Across Multiple Languages / Region
def forecast_language_arima(lang_pivot: pd.DataFrame, lang: str, test_days: int,
                             p_range=range(0, 3), q_range=range(0, 3)) -> dict:
    """Fit + forecast a per-language ARIMA model; returns forecast, actuals, metrics."""
    s = lang_pivot[lang].interpolate()
    tr, te = s.iloc[:-test_days], s.iloc[-test_days:]

    # Determine differencing order for this specific language (each language's
    # traffic can need a different d)
    d_lang = 0
    check = tr.copy()
    while d_lang < 3:
        p_adf = adfuller(check.dropna())[1]
        if p_adf < 0.05:
            break
        d_lang += 1
        check = tr.diff(d_lang).dropna()

    order, model, _ = grid_search_arima(tr, p_range=p_range, d_range=[d_lang], q_range=q_range)
    fc = model.forecast(steps=test_days)
    metrics = evaluate_forecast(te, fc, f"ARIMA{order} - {lang}")
    return {"lang": lang, "order": order, "d": d_lang, "train": tr, "test": te,
            "forecast": fc, "metrics": metrics}

ALL_LANGUAGES = sorted(set(lang_pivot.columns) - NON_LANGUAGE_BUCKETS)
print("Languages to forecast:", ALL_LANGUAGES)

multi_lang_results = [forecast_language_arima(lang_pivot, lang, TEST_DAYS) for lang in ALL_LANGUAGES]

multi_lang_metrics = pd.DataFrame([r["metrics"] for r in multi_lang_results]).round(3)
display(multi_lang_metrics)

n = len(multi_lang_results)
fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
axes = np.atleast_1d(axes)
for ax, r in zip(axes, multi_lang_results):
    ax.plot(r["train"].index[-60:], r["train"].iloc[-60:], label="Train (last 60d)", color="grey")
    ax.plot(r["test"].index, r["test"], label="Actual", color="black")
    ax.plot(r["test"].index, r["forecast"], label="Forecast", color="tab:red", linestyle="--")
    ax.set_title(f"{r['lang']} — ARIMA{r['order']} (MAPE={r['metrics']['MAPE']:.1f}%)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
