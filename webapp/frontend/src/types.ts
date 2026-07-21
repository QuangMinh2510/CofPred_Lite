export interface Meta {
  project: string;
  target: string;
  raw_rows: number;
  series_rows: number;
  date_min: string;
  date_max: string;
  deployed_models: string[];
  deployed_horizons: number[];
  research_models: number;
  research_horizons: number[];
}

export interface PricePoint {
  date: string;
  value: number;
}

export interface ForecastPoint {
  date: string;
  forecast: number;
  lower_50: number;
  upper_50: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
}

export interface ForecastPayload {
  model: string;
  horizon: number;
  data_as_of: string;
  forecast_anchor_date: string;
  anchor_price: number;
  point_forecast: number;
  rmse: number;
  source: string;
  history: PricePoint[];
  forecast: ForecastPoint[];
  disclaimer: string;
}

export interface ComparisonModel {
  family: string;
  model: string;
  mse: number | null;
  delta_vs_rw_pct: number | null;
}

