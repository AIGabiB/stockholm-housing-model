import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_percentage_error,root_mean_squared_error
import numpy as np
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import joblib

# self created files
from src.data_manipulation.data_utils import clean_data
from src.data_manipulation.feature_enginnering import engineer_features

data = pd.read_parquet("data/raw_synthetic.parquet")

cleaned_data = clean_data(data)
engineered_data = engineer_features(cleaned_data)

x = engineered_data.drop(columns=["Sale Price"],errors="ignore")
y = engineered_data["Sale Price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
numerical_cols = make_column_selector(dtype_include=["number"])

binary_cols = ["Elevator", "Balcony", "Heating Included", "New Construction", "Housing Type"]                          


preprocessor = ColumnTransformer(transformers=[
    ("num",StandardScaler(),numerical_cols),
    ("catefew",OneHotEncoder(handle_unknown="ignore"),binary_cols)
    ],remainder="passthrough")


pipe = Pipeline(steps=[
    ("scale_data", preprocessor),
    ("model", Ridge(alpha=1.0))
])

pipe.fit(x_train,y_train)   

y_pred_log = pipe.predict(x_test)


y_test_original = np.exp(y_test)
y_pred_original = np.exp(y_pred_log) 

r2 = r2_score(y_test, y_pred_log)

rmse = root_mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original,y_pred_original)


linear_regression_metrics = pd.DataFrame({
    "R²":[r2],
    "MAE":[mae],
    "RMSE":[rmse],
    "MAPE":[mape]
})

importances = pipe.named_steps["model"].coef_

features_names = pipe.named_steps["scale_data"].get_feature_names_out()

lin_importance = pd.Series(index=features_names,data=importances).abs().sort_values(ascending=True)
lin_importance.index = lin_importance.index.str.split("__").str[-1]

feature_importance_grouped = (
    lin_importance
    .groupby(lambda x: x.split("_")[0])
    .sum()
    .sort_values(ascending=False)
)

results_ridge = {
    "model": pipe,
    "y_pred": y_pred_original,
    "y_test": y_test_original,
    "metrics": linear_regression_metrics,
    "feature_importance": feature_importance_grouped       
    }

joblib.dump(results_ridge, "saved_models/ridge_results.pkl")
