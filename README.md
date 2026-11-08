
# Predicting Weekly Earnings Using Machine Learning

### Evidence from the December 2024 Current Population Survey (CPS)

**Author:** Anaeli Pinto  
**Field:** Applied Econometrics · Labor Economics · Machine Learning  
**Tools:** Python, Jupyter Notebook, pandas, NumPy, scikit-learn, Matplotlib

## Overview

This independent research project examines the relative predictive importance of education, hours worked, and demographic characteristics in explaining individual weekly earnings in the United States. Using December 2024 Current Population Survey (CPS) microdata, I estimate and tune a Random Forest regression model to assess which variables contribute most to earnings predictions.

**Research question:** To what extent do education and hours worked dominate demographic characteristics in predicting weekly earnings?

## Data

- **Source:** U.S. Census Bureau / Bureau of Labor Statistics, Current Population Survey, December 2024 basic monthly public-use file (`dec24pub.csv`).
- **Analytical sample:** 9,675 observations after filtering for positive reported weekly earnings and valid work hours, and removing missing values in the selected fields.
- **Target variable:** Natural logarithm of weekly earnings, derived from `PTERNWA` (converted from its reported cent-based units to dollars).
- **Model predictors:** Educational attainment (`PEEDUCA`), usual hours worked (`PEHRUSL1`), age (`PRTAGE`), sex (`PESEX`), marital status (`PEMARITL`), race (`PTDTRACE`), and citizenship status (`PRCITSHP`).

The notebook also uses actual hours worked (`PEHRACT1`) when filtering the sample, although the final regression uses *usual* hours worked as its hours predictor.

## Methodology

1. **Data preparation:** Select CPS variables, restrict the sample to valid positive earnings and work-hour observations, drop missing values, convert weekly earnings to dollars, and log-transform the earnings outcome.
2. **Exploratory analysis:** Calculate descriptive statistics and visualize earnings distributions, earnings by education, and the relationship between earnings and hours worked.
3. **Model selection:** Fit a `RandomForestRegressor` and tune hyperparameters with a parameter grid and shuffled **five-fold cross-validation** (`random_state=4465`).
4. **Evaluation:** Compare cross-validated root mean squared error (RMSE) across candidate configurations and inspect the fitted model's feature importance scores and actual-versus-predicted values.

## Results

| Metric | Result |
|---|---:|
| Cleaned observations | 9,675 |
| Best number of trees (`n_estimators`) | 200 |
| Best maximum tree depth (`max_depth`) | 10 |
| Best minimum split size (`min_samples_split`) | 10 |
| Best minimum leaf size (`min_samples_leaf`) | 1 |
| Best `max_features` | `sqrt` |
| **Mean five-fold cross-validation RMSE** | **0.603** |
| Full-sample fitted RMSE | 0.562 |

*RMSE values are measured on the natural-log weekly earnings scale. The full-sample fitted RMSE is an in-sample statistic, not a separate out-of-sample test score.*

### Predictor importance

| Predictor | Random Forest feature importance |
|---|---:|
| Usual hours worked | 52.2% |
| Education | 22.2% |
| Age | 14.1% |
| Marital status | 4.8% |
| Sex | 3.8% |
| Race | 1.4% |
| Citizenship | 1.4% |

In this fitted model, **usual hours worked and educational attainment are the strongest predictors** of weekly earnings, followed by age. The other included demographic characteristics receive lower feature-importance scores.

These scores describe predictive contributions within this model and sample; they **do not establish causal effects** or imply that demographic factors are unimportant in the labor market.

## Repository contents

- `weekly_earnings_prediction.ipynb` — research question, data preparation, descriptive statistics, visualizations, model tuning, and results.
- `README.md` — project summary and key findings.

## Reproducing the analysis

1. Download the December 2024 CPS public-use CSV (`dec24pub.csv`) from the official CPS data source.
2. Place `dec24pub.csv` in the same folder as the notebook.
3. Install the required Python packages:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

4. Open `weekly_earnings_prediction.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order.

The CSV is not included in this repository. The notebook expects the filename `dec24pub.csv` in its working directory.

## Limitations

- The analysis uses a single monthly CPS sample and a selected set of predictors; results may differ for other periods or model specifications.
- Random Forest feature importance measures predictive usefulness rather than causal influence.
- The notebook uses the CPS-coded predictor values directly, without a separate categorical encoding strategy.
- Hyperparameter selection and reported cross-validation performance are based on the same cross-validation procedure; no independent holdout test set is reported.

## Conclusion

The December 2024 CPS analysis suggests that usual hours worked and educational attainment contribute more to Random Forest predictions of weekly earnings than the demographic variables included in this specification. The project illustrates how supervised machine learning can be used to investigate empirical questions in labor economics while highlighting the distinction between predictive importance and causal interpretation.
