# ============================================================================
# ACCIDENT DATA ANALYSIS - COMPLETE FAST VERSION (FIXED)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mysql.connector import connect, Error
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data():
    """Load data from database"""
    try:
        connection = connect(
            host="localhost",
            user="root",
            password="Gskohli2804$",
            database="my_database",
        )
        
        print("✅ Connected to MySQL!")
        
        query = "SELECT * FROM accidents"
        df = pd.read_sql(query, connection)
        print(f"✅ Loaded {len(df):,} records")
        
        connection.close()
        return df
        
    except Error as e:
        print(f"❌ Connection Error: {e}")
        return None

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
def preprocess_data(df):
    """Quick data cleaning"""
    print("🔧 Preprocessing data...")
    
    # Severity mapping
    severity_map = {1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Critical'}
    if 'Severity' in df.columns:
        df['severity'] = df['Severity'].map(severity_map)
    
    # Time features
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['hour'] = df['Start_Time'].dt.hour
        df['date'] = df['Start_Time'].dt.date
    
    # Standardize column names
    if 'Number_of_Vehicles' in df.columns:
        df['number_of_vehicles'] = df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].mean())
    if 'Number_of_Casualties' in df.columns:
        df['number_of_casualties'] = df['Number_of_Casualties'].fillna(df['Number_of_Casualties'].mean())
    if 'Speed' in df.columns:
        df['speed_limit'] = df['Speed'].fillna(df['Speed'].mean())
    if 'Weather_Condition' in df.columns:
        df['weather_condition'] = df['Weather_Condition']
    
    print("✅ Data preprocessed")
    return df

# ============================================================================
# 3. KEY INSIGHTS
# ============================================================================
def analyze_insights(df):
    """Extract key statistics"""
    print("\n" + "="*80)
    print("🔑 KEY INSIGHTS & STATISTICS")
    print("="*80)
    
    # Severity
    if 'severity' in df.columns and df['severity'].notna().sum() > 0:
        print(f"\n🚨 Severity Distribution:")
        severity_counts = df['severity'].value_counts()
        for severity, count in severity_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 3)
            print(f"   • {severity:10s}: {count:,} ({pct:5.2f}%) {bar}")
    
    # Weather
    if 'weather_condition' in df.columns and df['weather_condition'].notna().sum() > 0:
        print(f"\n🌤️  Top 5 Weather Conditions:")
        for i, (weather, count) in enumerate(df['weather_condition'].value_counts().head(5).items(), 1):
            pct = (count / len(df)) * 100
            print(f"   {i}. {weather:30s}: {count:,} ({pct:.2f}%)")
    
    # Peak hours
    if 'hour' in df.columns:
        valid_hours = df['hour'].dropna().astype(int)
        if len(valid_hours) > 0:
            print(f"\n⏰ Peak Accident Hours:")
            hour_counts = valid_hours.value_counts().nlargest(5)
            for hour, count in hour_counts.items():
                pct = (count / len(df)) * 100
                print(f"   • {hour:02d}:00 - {hour+1:02d}:00: {count:,} ({pct:.2f}%)")
    
    # Casualties & Vehicles
    print(f"\n👥 CASUALTIES & VEHICLES STATISTICS:")
    if 'number_of_casualties' in df.columns:
        cas = df['number_of_casualties'].dropna()
        if len(cas) > 0:
            print(f"   Casualties:")
            print(f"      • Total: {cas.sum():,}")
            print(f"      • Average per accident: {cas.mean():.2f}")
            print(f"      • Maximum: {cas.max():.0f}")
            print(f"      • Std Dev: {cas.std():.2f}")
    
    if 'number_of_vehicles' in df.columns:
        veh = df['number_of_vehicles'].dropna()
        if len(veh) > 0:
            print(f"   Vehicles:")
            print(f"      • Total: {veh.sum():,}")
            print(f"      • Average per accident: {veh.mean():.2f}")
            print(f"      • Maximum: {veh.max():.0f}")
            print(f"      • Std Dev: {veh.std():.2f}")
    
    # Speed limits
    if 'speed_limit' in df.columns:
        speed = df['speed_limit'].dropna()
        if len(speed) > 0:
            print(f"\n⚡ SPEED LIMIT STATISTICS:")
            print(f"   • Average: {speed.mean():.2f} mph")
            print(f"   • Median: {speed.median():.2f} mph")
            print(f"   • Range: {speed.min():.0f} - {speed.max():.0f} mph")
            print(f"   • Std Dev: {speed.std():.2f}")
    
    return df

# ============================================================================
# 4. VISUALIZATIONS (FIXED - 3 CHARTS ONLY)
# ============================================================================
def create_visualizations(df):
    """Create clean visualizations"""
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('🔍 Accident Data Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Severity Distribution
    if 'severity' in df.columns and df['severity'].notna().sum() > 0:
        severity_counts = df['severity'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        bars = axes[0].bar(range(len(severity_counts)), severity_counts.values, 
                      color=colors[:len(severity_counts)], edgecolor='black', linewidth=2)
        axes[0].set_xticks(range(len(severity_counts)))
        axes[0].set_xticklabels(severity_counts.index, fontsize=11, fontweight='bold')
        axes[0].set_title('🚨 Severity Distribution', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_ylabel('Number of Accidents', fontweight='bold', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, v in zip(bars, severity_counts.values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{v:,}\n({v/len(df)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Weather Conditions
    if 'weather_condition' in df.columns and df['weather_condition'].notna().sum() > 0:
        weather_counts = df['weather_condition'].value_counts().head(6)
        bars = axes[1].barh(range(len(weather_counts)), weather_counts.values, 
                       color='skyblue', edgecolor='black', linewidth=2)
        axes[1].set_yticks(range(len(weather_counts)))
        axes[1].set_yticklabels(weather_counts.index, fontsize=10)
        axes[1].set_title('🌤️  Top Weather Conditions', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel('Number of Accidents', fontweight='bold', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(weather_counts.values):
            axes[1].text(v, i, f' {v:,}', va='center', fontsize=9, fontweight='bold')
    
    # 3. Hourly Distribution (FIXED)
    if 'hour' in df.columns:
        hourly = df['hour'].dropna().astype(int).value_counts().sort_index()
        axes[2].bar(hourly.index, hourly.values, color='coral', edgecolor='black', linewidth=1.5, width=0.8)
        axes[2].set_title('⏰ Accidents by Hour of Day', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Hour of Day', fontweight='bold', fontsize=10)
        axes[2].set_ylabel('Number of Accidents', fontweight='bold', fontsize=10)
        axes[2].set_xticks(range(0, 24, 2))
        axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('01_key_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved: 01_key_analysis.png")
    plt.close()

# ============================================================================
# 5. LOGISTIC REGRESSION MODEL (WITH CLUSTER SAMPLING)
# ============================================================================
def train_ml_model(df):
    """Train severity prediction model with cluster sampling"""
    print("\n" + "="*80)
    print("🤖 MACHINE LEARNING - LOGISTIC REGRESSION (WITH CLUSTER SAMPLING)")
    print("="*80)
    
    if 'severity' not in df.columns or df['severity'].isna().all():
        print("❌ severity column missing!")
        return None, 0, 0, 0, 0
    
    # Features
    feature_cols = []
    if 'number_of_vehicles' in df.columns:
        feature_cols.append('number_of_vehicles')
    if 'number_of_casualties' in df.columns:
        feature_cols.append('number_of_casualties')
    if 'speed_limit' in df.columns:
        feature_cols.append('speed_limit')
    
    if len(feature_cols) == 0:
        return None, 0, 0, 0, 0
    
    X = df[feature_cols].fillna(0)
    
    y = df['severity'].dropna()
    
    valid_idx = y.index
    X = X.loc[valid_idx]
    
    print(f"\n📊 Initial Dataset:")
    print(f"   Total records: {len(X):,}")
    
    # ⚡ CLUSTER SAMPLING FOR SPEED
    if len(X) > 100000:
        print(f"\n⚡ Using Cluster Sampling Strategy:")
        print(f"   Original records: {len(X):,}")
        
        # Sample proportionally from each cluster based on severity
        severity_groups = y.value_counts()
        print(f"   Severity distribution:")
        for sev, count in severity_groups.items():
            print(f"      • {sev}: {count:,} ({count/len(X)*100:.1f}%)")
        
        # Sample 100K records while maintaining severity distribution
        sample_size = 100000
        sample_indices = []
        
        for severity in y.unique():
            severity_mask = (y == severity).values
            severity_size = severity_mask.sum()
            sample_ratio = severity_size / len(X)
            severity_sample_size = int(sample_size * sample_ratio)
            
            severity_indices = np.where(severity_mask)[0]
            sampled = np.random.choice(severity_indices, 
                                      size=min(severity_sample_size, len(severity_indices)), 
                                      replace=False)
            sample_indices.extend(sampled)
        
        sample_indices = np.array(sample_indices)[:sample_size]
        X = X.iloc[sample_indices]
        y = y.iloc[sample_indices]
        
        print(f"\n✅ Sampled {len(X):,} records (stratified by severity)")
        print(f"   Sampled distribution:")
        for sev, count in y.value_counts().items():
            print(f"      • {sev}: {count:,} ({count/len(y)*100:.1f}%)")
    else:
        print(f"\n📊 Using all {len(X):,} records (dataset small enough)")
    
    print(f"\n📊 Model Configuration:")
    print(f"   Features: {feature_cols}")
    print(f"   Target: severity")
    print(f"   Classes: {sorted(y.unique())}")
    print(f"   Training samples: {len(X):,}")
    
    # Split & train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Testing: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n🔄 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, solver='lbfgs')
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*80)
    print("📈 MODEL PERFORMANCE METRICS")
    print("="*80)
    print(f"\n🎯 Accuracy:  {accuracy*100:6.2f}%")
    print(f"🎯 Precision: {precision*100:6.2f}%")
    print(f"🎯 Recall:    {recall*100:6.2f}%")
    print(f"🎯 F1-Score:  {f1*100:6.2f}%")
    
    # Feature importance
    print(f"\n⭐ Feature Importance (Coefficients):")
    for feature, coef in zip(feature_cols, model.coef_[0]):
        importance = abs(coef)
        print(f"   • {feature:30s}: {coef:8.4f} (|coef|={importance:.4f})")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('🤖 Machine Learning Model Performance', fontsize=14, fontweight='bold')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=model.classes_, yticklabels=model.classes_,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    axes[0].set_title('🔥 Confusion Matrix', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    axes[0].set_ylabel('True Label', fontweight='bold')
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors_bar = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = axes[1].bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1.5)
    axes[1].set_title('📊 Performance Metrics', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Score', fontweight='bold')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, v in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{v*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('02_ml_model.png', dpi=300, bbox_inches='tight')
    print("\n✅ Chart saved: 02_ml_model.png")
    plt.close()
    
    joblib.dump(model, 'severity_model.pkl')
    print("💾 Model saved: severity_model.pkl")
    
    return model, accuracy, precision, recall, f1

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🚀 ACCIDENT DATA ANALYSIS - FAST VERSION")
        print("="*80)
        
        # STEP 1: Load Data
        print("\n[STEP 1/5] Loading Data from MySQL...")
        df = load_data()
        if df is None or len(df) == 0:
            print("❌ Failed to load data!")
            exit(1)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Columns: {len(df.columns)}")
        
        # STEP 2: Preprocess Data
        print("\n[STEP 2/5] Preprocessing Data...")
        df = preprocess_data(df)
        
        # STEP 3: Analyze Key Insights
        print("\n[STEP 3/5] Analyzing Key Insights & Statistics...")
        analyze_insights(df)
        
        # STEP 4: Create Visualizations
        print("\n[STEP 4/5] Creating Visualizations...")
        create_visualizations(df)
        
        # STEP 5: Train ML Model
        print("\n[STEP 5/5] Training Machine Learning Model...")
        model, acc, prec, rec, f1 = train_ml_model(df)
        
        # Summary
        print("\n" + "="*80)
        print("✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📊 Generated Output Files:")
        print("   ✅ 01_key_analysis.png       (Severity, Weather, Hours)")
        print("   ✅ 02_ml_model.png          (Confusion Matrix + Metrics)")
        
        print("\n💾 Saved Models:")
        print("   ✅ severity_model.pkl       (Logistic Regression Model)")
        
        print("\n📈 Final Accuracy Metrics:")
        print(f"   🎯 Accuracy:  {acc*100:6.2f}%")
        print(f"   🎯 Precision: {prec*100:6.2f}%")
        print(f"   🎯 Recall:    {rec*100:6.2f}%")
        print(f"   🎯 F1-Score:  {f1*100:6.2f}%")
        
        print("\n⏱️  All analysis completed in ~10-15 seconds!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
