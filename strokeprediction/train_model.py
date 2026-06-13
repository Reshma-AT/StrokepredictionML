import os
import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

def train_and_export():
    # 1. Initialize H2O cluster background server
    h2o.init(max_mem_size='12G', nthreads=4)
    
    # 2. Data Preprocessing (Brought over exactly from your notebook)
    df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
    df.bmi = df.bmi.fillna(-99)
    df.rename(columns={'Residence_type': 'residence_type'}, inplace=True)
    df['target'] = df.stroke
    df = df.drop(['stroke'], axis=1) 
    
    # Force hypertension and heart_disease to strings so pandas/h2o treats them as categorical
    df['hypertension'] = df['hypertension'].astype(str)
    df['heart_disease'] = df['heart_disease'].astype(str)
    
    features_num = ['age', 'avg_glucose_level', 'bmi']
    features_cat = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
    predictors = features_num + features_cat
    
    # 3. Load structured DataFrame into H2O Frame
    df_hex = h2o.H2OFrame(df)
    
    # Explicitly set all categorical predictors and the target as factors
    for col in features_cat:
        df_hex[col] = df_hex[col].asfactor()
    df_hex['target'] = df_hex['target'].asfactor()
    
    # 4. Define and Train Gradient Boosting Model
    fit_1 = H2OGradientBoostingEstimator(
        ntrees=100,
        max_depth=4,
        min_rows=10,
        learn_rate=0.01,
        sample_rate=1,
        col_sample_rate=0.7,
        nfolds=5,
        score_each_iteration=True,
        stopping_metric='auto',
        stopping_rounds=10,
        seed=999
    )
    
    print("Training the H2O Gradient Boosting model...")
    fit_1.train(x=predictors, y='target', training_frame=df_hex)
    
    # 5. Export Model for app.py production deployment
    os.makedirs('model', exist_ok=True)
    mojo_path = fit_1.download_mojo(path="model", get_genmodel_jar=True)
    print(f"Model successfully saved to: {mojo_path}")
    
    h2o.cluster().shutdown()

if __name__ == "__main__":
    train_and_export()