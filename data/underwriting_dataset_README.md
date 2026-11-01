# Underwriting risk demo — 10 historical property baselines + simulated signals

Created: 2026-11-01. Data span: November 2025–October 2026 (12 fully completed months).

## Files
- `underwriting_risk_demo_10_properties.xlsx`: dashboard, 10 properties, 120 monthly records, trends, dictionary, provenance.
- `underwriting_properties_10.csv`: 10 property rows.
- `underwriting_monthly_risk_120.csv`: 120 property-month rows (join on `property_id`).
- `underwriting_field_dictionary.csv`: schema/provenance of all fields.

## Real / public source (historical, NOT 2026)
The first ten records of the Kaggle *House Sales in King County, USA* dataset form the property baseline.
Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data
Exact source CSV: https://raw.githubusercontent.com/46n/king-county-house-sales-analysis/main/data/housing.csv
Only `latitude`, `longitude`, `year_built`, `zip_code`, `square_footage`, `historical_sale_price_usd`, `historical_sale_date`, and `original_kaggle_id` are directly from the original historical records. City/state are approximate geographic labels inferred from ZIP.

## Simulated / constructed
Every other metric, street-address placeholder, estimated home value, roof age, flood zone, distance to water, 12-month weather occurrence, climate trend, neighborhood indicator, construction inflation, labor index, risk score, underwriting decision label, and underwriting note is invented as a **test fixture**. No FEMA flood GIS query, NOAA weather extraction, geocoding, geospatial distance computation, valuation model, or insurance underwriting algorithm has been run. Demo addresses are intentionally *not* valid street addresses. Do not quote hazards as measured facts.

## Risk score definition (not an actuarial model)
Score is capped at 100, built as a weighted sum of synthetic flood categories, roof age, distance to water, annual simulated weather counts, heat/precipitation/wildfire trends, crime/density, and artificial construction inflation/labor indices. The formula is **fully editable in the Excel Properties tab**. Thresholds: Low 0–34, Moderate 35–59, High 60–79, Very High 80–100.

## Monthly trend construction
The 2025-11 through 2026-10 counts are random-but-seeded allocations of each property annual scenario count with seasonality. Each weather total adds up exactly to the property-level 12-month value. The monthly risk score is centered around the baseline property score and adjusted by monthly weather, subtracting annual mean event contribution, plus a small seasonal climate factor. Monthly formulas are editable in Excel.

## Reference sources for real production enrichment (NOT used for any risk fields)
NOAA Storm Events Database: https://www.ncei.noaa.gov/access/storm-events-database/
FEMA flood hazard maps: https://msc.fema.gov/portal/home
Real enrichment would also need roof/property inspection, updated property valuations, historical weather extraction, climate projections, population/crime statistics, and local construction/labor price series.
