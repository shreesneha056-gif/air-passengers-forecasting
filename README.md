# DataScience_MachineLearning----Forecasting----AirPassengers_forecasting_project
# Airline Passengers Time Series Forecasting (ARIMA & SARIMA)

This repository hosts a machine learning and statistical time series project focused on forecasting monthly international airline passenger traffic. Utilizing historical data, this case study implements classic parametric statistical modeling—ranging from baseline Autoregressive Integrated Moving Average (**ARIMA**) to Seasonal ARIMA (**SARIMA**)—to predict future passenger demand.

---

## 📂 Dataset Repository & Properties

The model consumes historical airline traffic configurations hosted in a single clean time-slice file.

### Dataset Profile (`AirPassengers.csv`)
* **Dimensions:** 144 Monthly observations spanning from **January 1949 to December 1960**.
* **Attributes:**
  - `Month`: Date identifier formatted as `YYYY-MM` (Set as the Datetime index).
  - `#Passengers`: Count of international airline passengers (thousands per month).
* **Characteristics:** Displays a prominent long-term linear upward trend combined with an expanding multiplicative seasonal variance peaking during summer vacation months.

---

## 🛠️ Time Series Engineering & Pipeline

The code inside `Forecasting_AirPassangers_ARIMA_SARIMA.ipynb` implements a rigid time series data engineering flow:

1. **Stationarity Stabilization:** Because the passenger variance expands over time, a **Log Transformation** is executed to stabilize variance.
2. **Differencing Tests:** Regular differencing ($d$) and Seasonal differencing ($D$) are evaluated alongside **Augmented Dickey-Fuller (ADF)** tests to strip out trends and seasonality to render the series stationary.
3. **ACF / PACF Analysis:** Autocorrelation and Partial Autocorrelation plots are used to isolate initial structural guesses for Autoregressive ($p, P$) and Moving Average ($q, Q$) orders.

---

## 🤖 Modeling Framework & Architecture

The forecasting notebook tests, scores, and isolates performance across two primary configurations:

### 1. Baseline ARIMA Model
* Evaluates non-seasonal boundaries capturing pure autoregressive lags and moving error corrections.
* Structurally limited when trying to generalize against the stark repeating annual peaks present in aviation cycles.

### 2. Seasonal SARIMA Model
* **Configuration:** Extends standard parameters to account for repeating cyclic behavior over a seasonal period of $S = 12$ months.
* **Optimization & Grid Search:** Executes parameter iterations to pinpoint the ideal combination of `(p, d, q) x (P, D, Q)12` that minimizes the Akaike Information Criterion (**AIC**).
* **Mathematical Inversion:** Since training runs over log-transformed data, the final forecast inputs are mapped through an exponential transformation to revert findings to their original passenger scales:
  ```python
  best_sarima_f_cast = np.exp(best_sarima.forecast(12))
