import pandas as pd
import numpy as np
import argparse
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main(pitch_type):
    # Define the pitch type groups
    pitch_type_map = {
        'fastball': ['Fastball', 'Sinker', 'TwoSeamFastBall', 'FourSeamFastBall', 'OneSeamFastBall'],
        'slider': ['Slider', 'Cutter'],
        'curveball': ['Curveball'],
        'offspeed': ['Changeup', 'ChangeUp', 'Splitter']
    }
    
    if pitch_type not in pitch_type_map:
        raise ValueError(f"Invalid pitch_type '{pitch_type}'. Choose from {list(pitch_type_map.keys())}.")
    
    selected_pitch_types = pitch_type_map[pitch_type]
    
    # File path to the CSV
    csv_path = '/Users/benjaminreinhard/Desktop/trackman.csv'
    
    print("Loading data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Data loaded with shape: {df.shape}")
    
    # Filter the DataFrame based on TaggedPitchType and PitchCall
    print("Filtering data based on pitch types and pitch calls...")
    df_filtered = df[
        df['TaggedPitchType'].isin(selected_pitch_types) &
        df['PitchCall'].isin(['FoulBallNotFieldable', 'FoulBall', 'StrikeSwinging', 'FoulBallFieldable', 'InPlay'])
    ].copy()
    print(f"Filtered data shape: {df_filtered.shape}")
    
    # Assign run values based on PitchCall and AutoHitType
    run_values = {
        "SwStr": 11.6,  # Multiplying by 100 to scale up the values
        "Foul": 3.5,
        "GroundBall": 5.8,
        "FlyBall": -14.1  # This value applies to FlyBall, LineDrive, and Popup
    }
    
    print("Mapping run values...")
    def map_run_values(row):
        if row['PitchCall'] in ['FoulBallNotFieldable', 'FoulBall', 'FoulBallFieldable']:
            return run_values['Foul']
        elif row['PitchCall'] == 'StrikeSwinging':
            return run_values['SwStr']
        elif row['PitchCall'] == 'InPlay':
            if row['AutoHitType'] == 'GroundBall':
                return run_values['GroundBall']
            elif row['AutoHitType'] in ['FlyBall', 'LineDrive', 'Popup']:
                return run_values['FlyBall']
        return np.nan  # Assign NaN if conditions are not met
    
    df_filtered['RunValue'] = df_filtered.apply(map_run_values, axis=1)
    
    # Drop rows with NaN RunValue
    df_filtered.dropna(subset=['RunValue'], inplace=True)
    print(f"Data shape after mapping run values: {df_filtered.shape}")
    
    # Define features to be used
    features = [
        'RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis',
        'RelHeight', 'RelSide', 'InducedVertBreak', 'HorzBreak',
        'PlateLocHeight', 'PlateLocSide'
    ]
    
    # Check for missing values in features
    print("Checking for missing values in features...")
    missing_values = df_filtered[features].isnull().sum()
    print(f"Missing values in features:\n{missing_values}")
    
    # Drop rows with any missing feature values
    df_filtered.dropna(subset=features, inplace=True)
    print(f"Data shape after dropping missing feature values: {df_filtered.shape}")
    
    # Prepare data for modeling
    X = df_filtered[features]
    y = df_filtered['RunValue']
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize models to compare
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv),
        'Lasso': LassoCV(alphas=np.logspace(-3, 1, 10), cv=cv, max_iter=10000),
        'ElasticNet': ElasticNetCV(alphas=np.logspace(-3, 1, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv, max_iter=10000)
    }
    
    # Fit models and evaluate
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} MSE: {mse:.5f}")
        print(f"{name} R2 Score: {r2:.5f}")
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'y_pred': y_pred
        }
        
        # Display coefficients
        coef = model.coef_
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        print(f"{name} Coefficients:")
        print(coef_df)
    
    # Choose the best model based on R2 score
    best_model_name = max(results, key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model based on R2 score: {best_model_name}")
    
    # Scale predictions to "Stuff Score"
    print("Calculating Stuff Scores...")
    y_pred = results[best_model_name]['y_pred']
    mean_stuff = 100
    std_stuff = 10
    stuff_scores = (y_pred - y_pred.mean()) / y_pred.std() * std_stuff + mean_stuff
    df_results = pd.DataFrame({'PitcherId': df_filtered.iloc[y_test.index].index, 'Stuff_Score': stuff_scores})
    
    # Display top and bottom 10 pitchers by Stuff Score
    print("\nTop 10 Pitchers by Stuff Score:")
    top_10 = df_results.nlargest(10, 'Stuff_Score')[['PitcherId', 'Stuff_Score']]
    print(top_10)
    
    print("\nBottom 10 Pitchers by Stuff Score:")
    bottom_10 = df_results.nsmallest(10, 'Stuff_Score')[['PitcherId', 'Stuff_Score']]
    print(bottom_10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Stuff Scores for different pitch types.")
    parser.add_argument('--pitch_type', type=str, required=True,
                        choices=['fastball', 'slider', 'curveball', 'offspeed'],
                        help="Type of pitch to analyze.")
    args = parser.parse_args()
    main(args.pitch_type)
