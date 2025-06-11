
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def load_model(model_path="model.pkl", features_path="features.pkl"):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def load_data(csv_file):
    return pd.read_csv(csv_file)

def evaluate_model(model, features_list, df, target_column='SalePrice'):
    df = df.select_dtypes(include='number').dropna()
    X = df[features_list]
    y = df[target_column]
    y_pred = model.predict(X)
    scores = {
        "R2 Score": r2_score(y, y_pred),
        "MSE": mean_squared_error(y, y_pred)
    }
    return scores

def predict_single(model, input_dict, features_list):
    df = pd.DataFrame([input_dict])
    df = df[features_list]  
    return model.predict(df)[0]
