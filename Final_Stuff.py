import pandas as pd
import numpy as np
import xgboost as xgb

def process_pitch_type(df, pitch_type, global_mean, global_std):
    pitch_type_map = {
        'fastball': ['Fastball', 'Sinker', 'TwoSeamFastBall', 'FourSeamFastBall', 'OneSeamFastBall'],
        'slider': ['Slider', 'Cutter'],
        'curveball': ['Curveball'],
        'offspeed': ['Changeup', 'ChangeUp', 'Splitter']
    }
    
    selected_pitch_types = pitch_type_map[pitch_type]
    
    df_filtered = df[
        df['TaggedPitchType'].isin(selected_pitch_types) &
        df['PitchCall'].isin(['FoulBallNotFieldable', 'FoulBall', 'StrikeSwinging', 'FoulBallFieldable', 'InPlay'])
    ].copy()
    
    df_filtered = df_filtered[df_filtered['PitcherThrows'] != 'Undefined']
    
    # Adjusting HorzRelAngle and SpinAxis for left-handed pitchers
    df_filtered['HorzRelAngle'] = np.where(df_filtered['PitcherThrows'] == 'Left', df_filtered['HorzRelAngle'] * -1, df_filtered['HorzRelAngle'])
    df_filtered['SpinAxis'] = np.where(df_filtered['PitcherThrows'] == 'Left', 360 - df_filtered['SpinAxis'], df_filtered['SpinAxis'])
    
    # Use absolute value of HorzBreak
    df_filtered['HorzBreak'] = df_filtered['HorzBreak'].abs()
    
    # Mapping run values
    run_values = {
        "SwStr": 11.3,
        "Foul": 4.0,
        "GroundBall": 6.1,
        "Popup": 23.3,
        "FlyBall": -20.1,
    }
    
    def map_run_values(row):
        if row['PitchCall'] in ['FoulBallNotFieldable', 'FoulBall', 'FoulBallFieldable']:
            return run_values['Foul']
        elif row['PitchCall'] == 'StrikeSwinging':
            return run_values['SwStr']
        elif row['PitchCall'] == 'InPlay':
            if row['AutoHitType'] == 'GroundBall':
                return run_values['GroundBall']
            elif row['AutoHitType'] in ['FlyBall', 'LineDrive']:
                return run_values['FlyBall']
            elif row['AutoHitType'] == 'Popup':
                return run_values['Popup']
        return np.nan
    
    df_filtered['RunValue'] = df_filtered.apply(map_run_values, axis=1)
    df_filtered.dropna(subset=['RunValue'], inplace=True)
    
    # Normalize and scale RunValue
    df_filtered['RunValue'] = ((df_filtered['RunValue'] - global_mean) / global_std) * 10 + 100
    
    if pitch_type == 'fastball':
        df_filtered['RelHeight_InducedVertBreak'] = df_filtered['RelHeight'] * df_filtered['InducedVertBreak']
    else:
        df_filtered['RelHeight_InducedVertBreak'] = 0
    
    features = [
        'RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis',
        'RelHeight', 'RelSide', 'InducedVertBreak', 'HorzBreak'
    ]
    
    if pitch_type == 'fastball':
        features.append('RelHeight_InducedVertBreak')
    
    df_filtered = pd.get_dummies(df_filtered, columns=['PitcherThrows'], drop_first=True)
    all_features = features + [col for col in df_filtered.columns if 'PitcherThrows_' in col]
    
    # Drop rows with any NaN values in the selected features
    df_filtered.dropna(subset=all_features, inplace=True)
    
    return df_filtered[all_features], df_filtered['RunValue'], all_features, df_filtered

def calculate_stuff_plus(X_new, feature_importances, pitch_type, features):
    if pitch_type != 'fastball':
        if 'RelHeight_InducedVertBreak' in features:
            index = features.index('RelHeight_InducedVertBreak')
            feature_importances[index] = 0.0
    
    raw_stuff_plus_scores = np.dot(X_new, feature_importances)
    
    # Scale Stuff+ scores to have mean 100 and std dev 10
    mean_stuff_plus = np.mean(raw_stuff_plus_scores)
    std_stuff_plus = np.std(raw_stuff_plus_scores)
    scaled_stuff_plus_scores = 100 + (raw_stuff_plus_scores - mean_stuff_plus) / std_stuff_plus * 10
    
    return scaled_stuff_plus_scores

def main():
    csv_path = '/Users/benjaminreinhard/Desktop/trackman.csv'
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    pitch_types = ['fastball', 'slider', 'curveball', 'offspeed']
    
    all_importances = []
    stuff_plus_stats = []
    all_run_values = []

    # First, gather all run values to calculate global mean and std dev
    for pitch_type in pitch_types:
        _, run_values, _, _ = process_pitch_type(df, pitch_type, global_mean=0, global_std=1)
        all_run_values.extend(run_values)
    
    global_mean = np.mean(all_run_values)
    global_std = np.std(all_run_values)

    print(f"Global Mean of RunValue: {global_mean}")
    print(f"Global Std Dev of RunValue: {global_std}")

    # Process each pitch type with scaled run values
    for pitch_type in pitch_types:
        X, y, features, df_filtered = process_pitch_type(df, pitch_type, global_mean, global_std)

        # Print average metrics for the pitch type
        print(f"Average metrics for {pitch_type}:")
        print(df_filtered[features].mean())

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_model.fit(X, y)
        
        # Get initial feature importances
        importances = xgb_model.feature_importances_
        
        if len(importances) != len(features):
            raise ValueError("Mismatch between the number of features and the number of importances")

        # Calculate Stuff+ scores for the pitch type
        stuff_plus_scores = calculate_stuff_plus(X, importances, pitch_type, features)
        
        mean_stuff_plus = np.mean(stuff_plus_scores)
        std_stuff_plus = np.std(stuff_plus_scores)
        
        print(f"{pitch_type.capitalize()} Stuff+ Mean: {mean_stuff_plus}, Std Dev: {std_stuff_plus}")
        importance_df = pd.DataFrame({
            'Feature': features,
            f'{pitch_type.capitalize()} Importance': importances
        })
        
        all_importances.append(importance_df)
        stuff_plus_stats.append({
            'Pitch Type': pitch_type,
            'Mean Stuff+': mean_stuff_plus,
            'Std Dev Stuff+': std_stuff_plus
        })

    if all_importances:
        final_importances_df = pd.concat(all_importances, axis=1)
        final_importances_df = final_importances_df.loc[:, ~final_importances_df.columns.duplicated()]
        
        final_importances_csv = '/Users/benjaminreinhard/Desktop/feature_importances.csv'
        final_importances_df.to_csv(final_importances_csv, index=False)
        print(f"Feature importances saved to {final_importances_csv}")

    print("\nOverall Stuff+ Statistics:")
    for stat in stuff_plus_stats:
        print(f"{stat['Pitch Type'].capitalize()} - Mean Stuff+: {stat['Mean Stuff+']}, Std Dev Stuff+: {stat['Std Dev Stuff+']}")

if __name__ == "__main__":
    main()
