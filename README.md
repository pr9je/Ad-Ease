# Ad Ease — Wikipedia Page Views Forecasting
### Time Series Business Case 

**Author:** Data Science Team
**Objective:** Forecast daily Wikipedia page views across languages to help advertisers
optimize ad placement, campaign timing, and budget allocation.

## Problem Statement

**Business objective.** Ad Ease provides ad infrastructure that helps clients place
advertisements on Wikipedia pages in the languages and regions that matter to them.
Advertisers pay for placements, and the return on that spend depends heavily on *when*
and *where* (which language/page) traffic will be high.

**ML / forecasting objective.** Given ~145,000 Wikipedia pages with 550 days of daily
view counts (across multiple languages, access types, and access origins), build time
series models that forecast future daily views — first aggregated at the **language**
level (the grain advertisers actually plan around), then extended with an English-only
campaign indicator (exogenous variable) to quantify campaign lift.

**Why this matters commercially**
- **Ad placement:** high-traffic windows are worth more; forecasts let Ad Ease price and
  place ads dynamically instead of reactively.
- **Campaign timing:** knowing the expected organic trend lets clients schedule
  campaigns to amplify (not fight) natural demand.
- **Customer acquisition:** language-level forecasts show clients which markets are
  growing, guiding where to acquire new advertisers.
- **Revenue:** more accurate forecasts reduce wasted spend on low-traffic windows and
  protect against under-serving high-traffic ones, directly improving ROI for clients
  and retention for Ad Ease.

  
**Expected business impact:** a forecasting pipeline with MAPE in the 4–8% range is
accurate enough to support automated bidding/placement decisions and campaign scheduling
recommendations with confidence.

