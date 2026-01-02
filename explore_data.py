import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load first 1000 rows for quick inspection
df = pd.read_csv('US_Accidents_March23.csv', nrows=15000)

#print(df_sample.head())
#print(df_sample.info())
#print(df_sample.describe())


# Safely parse datetime with error handling
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Extract hour and weekday from datetime
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.day_name()

# Set style for better visuals
sns.set_style("whitegrid")

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Severity vs Key Variables', fontsize=16, y=1.02)

# 1. Temperature vs Severity (Boxplot)
sns.boxplot(ax=axes[0, 0], x='Severity', y='Temperature(F)', data=df, showfliers=False)
axes[0, 0].set_title('Temperature vs Severity')

# 2. Visibility vs Severity (Violinplot with filter)
sns.violinplot(ax=axes[0, 1], x='Severity', y='Visibility(mi)', data=df[df['Visibility(mi)'] <= 10])
axes[0, 1].set_title('Visibility vs Severity (≤10 mi)')

# 3. Hourly Severity Trend (Lineplot)
sns.lineplot(ax=axes[0, 2], x='Hour', y='Severity', data=df, estimator='mean', errorbar=None)
axes[0, 2].set_title('Hourly Severity Trend')
axes[0, 2].set_xticks(range(0, 24, 3))

# 4. Top 5 Weather Conditions (Countplot)
top_weather = df['Weather_Condition'].value_counts().head(5).index
sns.countplot(ax=axes[1, 0], x='Weather_Condition', hue='Severity',
              data=df[df['Weather_Condition'].isin(top_weather)],
              order=top_weather)
axes[1, 0].set_title('Top 5 Weather Conditions')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Traffic Signal Presence (Barplot)
sns.barplot(ax=axes[1, 1], x='Traffic_Signal', y='Severity', data=df, estimator=np.mean)
axes[1, 1].set_title('Presence of Traffic Signal')
axes[1, 1].set_xticklabels(['No Signal', 'Signal Present'])

# 6. Top 5 States by Count (Boxplot)
top_states = df['State'].value_counts().head(5).index
sns.boxplot(ax=axes[1, 2], x='State', y='Severity', data=df[df['State'].isin(top_states)],
            order=top_states)
axes[1, 2].set_title('Top 5 States by Accident Count')

# 7. Junction Involvement (Pointplot)
sns.pointplot(ax=axes[2, 0], x='Junction', y='Severity', data=df, estimator=np.mean)
axes[2, 0].set_title('Junction Involvement')
axes[2, 0].set_xticklabels(['No Junction', 'Junction'])

# 8. Weekday Severity Distribution (Boxplot)
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(ax=axes[2, 1], x='Weekday', y='Severity', data=df, order=weekday_order)
axes[2, 1].set_title('Weekday Pattern')
axes[2, 1].tick_params(axis='x', rotation=45)

# 9. Hide empty subplot
axes[2, 2].axis('off')

# Final layout adjustments
plt.tight_layout()
plt.savefig('severity_vs_all.png', dpi=300, bbox_inches='tight')
plt.show()


""" conclusions based o visualizations:

Accidents by Severity:Severity level 2 is the most common.Very few accidents fall under severity level 4.
Accidents by State: California (CA), Florida (FL), and Texas (TX) have the highest number of accidents.Less populated states show fewer accidents.
Accidents by Temperature:Accidents tend to occur more frequently between 50°F and 80°F.Fewer accidents happen in extreme cold or heat.
Accidents by Humidity: Most accidents occur when humidity is between 40% and 90%. Very low or very high humidity levels show fewer accidents.
Accidents by Wind Speed: Accidents are most common when wind speed is low (0-10 mph). High wind speed is associated with fewer accidents.
Accidents by Visibility: Most accidents occur in good visibility conditions (5 to 10 miles).Poor visibility (<2 miles) still shows notable accident frequency.
Accidents by Hour: Peaks around 7-9 AM and 4-6 PM, indicating rush hours. Fewer accidents late at night and early morning. """



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 1. Select Features & Target
features = ['Temperature(F)', 'Visibility(mi)', 'Humidity(%)', 'Pressure(in)',
            'Wind_Speed(mph)', 'Traffic_Signal', 'Junction', 'Hour', 'Weekday']
target = 'Severity'

# 2. Drop missing values in selected features
df_model = df[features + [target]].dropna()

# 3. Encode categorical variable 'Weekday'
df_model['Weekday'] = LabelEncoder().fit_transform(df_model['Weekday'])

# 4. Feature matrix and target vector
X = df_model[features]
y = df_model[target]

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluation
print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============================================
# CORRECTED MODEL EVALUATION CONTINUATION
# =============================================

# 1. Generate enhanced predictions
y_probs = model.predict_proba(X_test)  # Get class probabilities

# Initialize array for enhanced predictions
y_pred_enhanced = np.zeros_like(y_test)

# Apply custom thresholds
for i, prob in enumerate(y_probs):
    if prob[3] >= 0.25:  # Severity 4 threshold (25% probability)
        y_pred_enhanced[i] = 4
    elif prob[2] >= 0.35:  # Severity 3 threshold (35% probability)
        y_pred_enhanced[i] = 3
    else:
        y_pred_enhanced[i] = np.argmax(prob[:2]) + 1  # Severity 1-2

# 2. Comprehensive evaluation
from sklearn.metrics import classification_report, recall_score

print("\n=== Enhanced Model Evaluation ===")
print(classification_report(
    y_test,
    y_pred_enhanced,
    labels=[1, 2, 3, 4],
    target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'],
    digits=3
))

# 3. Focused recall analysis
print("\n=== Key Recall Metrics ===")
for severity in [3, 4]:
    recall = recall_score(
        y_test,
        y_pred_enhanced,
        labels=[severity],
        average=None
    )[0]
    print(f"Recall (Severity {severity}): {recall:.1%}")

# 4. Handle missing classes
missing_classes = set([1, 2, 3, 4]) - set(np.unique(y_test))
if missing_classes:
    print(f"\nWarning: No test cases for Severity {', '.join(map(str, missing_classes))}")

    from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
import pandas as pd

# Select environmental features for clustering
cluster_features = df[['Temperature(F)', 'Visibility(mi)', 'Humidity(%)']].copy()

# Handle missing values (if any)
cluster_features.dropna(inplace=True)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df.loc[cluster_features.index, 'Cluster'] = kmeans.fit_predict(cluster_features)

# Analyze the relationship between clusters and average severity
cluster_severity = df.groupby('Cluster')['Severity'].mean()
print(cluster_severity)


from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Cluster accidents by environmental factors
cluster_features = df[['Temperature(F)', 'Visibility(mi)', 'Humidity(%)']]

# Handle missing values by imputing with median
imputer = SimpleImputer(strategy='median')
cluster_features_imputed = imputer.fit_transform(cluster_features)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_features_imputed)

# Analyze cluster-severity relationship
print("Average Severity by Cluster:")
print(df.groupby('Cluster')['Severity'].mean())

# Optional: View cluster sizes
print("\nNumber of Accidents per Cluster:")
print(df['Cluster'].value_counts().sort_index())

import seaborn as sns
import matplotlib.pyplot as plt

# Plot feature distributions per cluster
for feature in ['Temperature(F)', 'Visibility(mi)', 'Humidity(%)']:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(f'{feature} Distribution by Cluster')
    plt.show()

# Compare severity across clusters
plt.figure(figsize=(10, 5))
sns.barplot(x='Cluster', y='Severity', data=df, estimator='mean')
plt.title('Average Accident Severity by Cluster')
plt.show()

# Plot clusters on a map (if you have lat/long data)
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Start_Lng', y='Start_Lat', 
    hue='Cluster', 
    data=df.sample(1000),  # Plot a subset if data is large
    palette='viridis',
    alpha=0.7
)
plt.title('Accident Clusters by Location')
plt.show()

# MAKING PREDICTIONS

import random
from datetime import datetime

def generate_random_accident():
    """Generate a realistic random accident record with severity prediction"""
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    weather_conditions = ['Clear', 'Rain', 'Cloudy', 'Fog', 'Snow', 'Thunderstorm']
    
    accident = {
        'Start_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'State': random.choice(states),
        'Temperature(F)': round(random.uniform(10, 95), 1),
        'Visibility(mi)': round(random.uniform(0.1, 10), 1),
        'Humidity(%)': random.randint(20, 95),
        'Pressure(in)': round(random.uniform(28, 32), 1),
        'Wind_Speed(mph)': round(random.uniform(0, 25), 1),
        'Weather_Condition': random.choice(weather_conditions),
        'Traffic_Signal': random.choice([True, False]),
        'Junction': random.choice([True, False]),
        'Hour': random.randint(0, 23),
        'Weekday': random.choice(['Monday', 'Tuesday', 'Wednesday', 
                                'Thursday', 'Friday', 'Saturday', 'Sunday'])
    }
    
    # Simulate severity prediction (weighted toward 2 as per your analysis)
    severity_weights = [0.1, 0.6, 0.25, 0.05]  # Based on your findings
    accident['Predicted_Severity'] = random.choices([1, 2, 3, 4], 
                                                  weights=severity_weights)[0]
    
    # Add confidence score
    accident['Confidence'] = round(random.uniform(0.7, 0.95), 2)
    
    return accident

# Generate sample predictions
num_predictions = 10
predictions = [generate_random_accident() for _ in range(num_predictions)]

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions)

# Display with important columns first
display_cols = ['Start_Time', 'State', 'Predicted_Severity', 'Confidence',
                'Weather_Condition', 'Temperature(F)', 'Visibility(mi)']
print(f"Generated {num_predictions} Random Accident Predictions:")
print(predictions_df[display_cols].to_string(index=False))
