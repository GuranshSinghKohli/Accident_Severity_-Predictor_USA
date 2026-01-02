import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


KAGGLE_DATASET_URL = "https://www.kaggle.com/code/jingzongwang/usa-car-accidents-severity-prediction"


print("=" * 80)
print("📥 IMPORT ACCIDENT DATA TO MYSQL")
print("=" * 80)


# ============================================================================
# 1. DATABASE CONNECTION CONFIGURATION
# ============================================================================
print("\n📡 Connecting to MySQL Database...")

db_config = {
    'host': '127.0.0.1',  # Use IP instead of localhost
    'user': 'root',
    'password': 'Gskohli2804$',
    'database': 'my_database',
    'port': 3306,
    'autocommit': True
}

# ============================================================================
# 2. CREATE CONNECTION
# ============================================================================
def create_connection():
    """Create a connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✅ Successfully connected to MySQL Server version {db_info}")
            return connection
    except Error as e:
        print(f"❌ Error while connecting to MySQL: {e}")
        print("⚠️ Make sure MySQL is running and password is correct!")
        return None

# ============================================================================
# 3. LOAD DATA FROM CSV
# ============================================================================
def load_csv_data():
    """Load accident data from CSV file"""
    print("\n📂 Loading CSV data...")
    
    try:
        df = pd.read_csv('US_Accidents_March23.csv')
        print(f"✅ Loaded {len(df)} accident records from CSV")
        print(f"✅ Columns available: {df.shape[1]}")
        print(f"✅ First few columns: {df.columns.tolist()[:5]}")
        return df
    except FileNotFoundError:
        print("❌ CSV file 'US_Accidents_March23.csv' not found!")
        print("📍 Make sure it's in the same folder as this script")
        return None
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

# ============================================================================
# 4. CLEAN AND PREPARE DATA
# ============================================================================
def prepare_data(df):
    """Clean and prepare data for database insertion"""
    print("\n🔧 Preparing data...")
    
    # Select relevant columns (will handle missing columns gracefully)
    df_prep = df.copy()
    
    # Rename columns to match our table structure
    column_mapping = {
        'Start_Time': 'datetime',
        'Location': 'location',
        'Start_Lat': 'latitude',
        'Start_Lng': 'longitude',
        'Weather_Condition': 'weather_condition',
        'Road_Type': 'road_type',
        'Severity': 'severity'
    }
    
    # Rename only columns that exist
    available_mapping = {k: v for k, v in column_mapping.items() if k in df_prep.columns}
    df_prep = df_prep.rename(columns=available_mapping)
    
    # Handle datetime
    if 'datetime' in df_prep.columns:
        df_prep['datetime'] = pd.to_datetime(df_prep['datetime'], errors='coerce')
        df_prep['date'] = df_prep['datetime'].dt.date
        df_prep['time'] = df_prep['datetime'].dt.time
    else:
        df_prep['date'] = datetime.now().date()
        df_prep['time'] = datetime.now().time()
    
    # Create/assign values for required columns - FIXED VERSION
    if 'location' in df_prep.columns:
        df_prep['location'] = df_prep['location'].fillna('Unknown').astype(str).str[:255]
    else:
        df_prep['location'] = 'Unknown'
    
    if 'latitude' in df_prep.columns:
        df_prep['latitude'] = pd.to_numeric(df_prep['latitude'], errors='coerce')
    else:
        df_prep['latitude'] = np.nan
    
    if 'longitude' in df_prep.columns:
        df_prep['longitude'] = pd.to_numeric(df_prep['longitude'], errors='coerce')
    else:
        df_prep['longitude'] = np.nan
    
    if 'weather_condition' in df_prep.columns:
        df_prep['weather_condition'] = df_prep['weather_condition'].fillna('Unknown').astype(str).str[:100]
    else:
        df_prep['weather_condition'] = 'Unknown'
    
    if 'road_type' in df_prep.columns:
        df_prep['road_type'] = df_prep['road_type'].fillna('Unknown').astype(str).str[:100]
    else:
        df_prep['road_type'] = 'Unknown'
    
    if 'severity' in df_prep.columns:
        df_prep['severity'] = df_prep['severity'].fillna('Low').astype(str).str[:50]
    else:
        df_prep['severity'] = 'Low'
    
    # Generate random values if columns don't exist
    if 'speed_limit' not in df_prep.columns:
        df_prep['speed_limit'] = np.random.randint(25, 75, len(df_prep))
    else:
        df_prep['speed_limit'] = pd.to_numeric(df_prep['speed_limit'], errors='coerce').fillna(45).astype(int)
    
    if 'number_of_vehicles' not in df_prep.columns:
        df_prep['number_of_vehicles'] = np.random.randint(1, 8, len(df_prep))
    else:
        df_prep['number_of_vehicles'] = pd.to_numeric(df_prep['number_of_vehicles'], errors='coerce').fillna(2).astype(int)
    
    if 'number_of_casualties' not in df_prep.columns:
        df_prep['number_of_casualties'] = np.random.randint(0, 5, len(df_prep))
    else:
        df_prep['number_of_casualties'] = pd.to_numeric(df_prep['number_of_casualties'], errors='coerce').fillna(0).astype(int)
    
    df_prep['description'] = 'Accident reported via traffic monitoring system'
    
    # Select only needed columns
    columns_needed = ['date', 'time', 'location', 'latitude', 'longitude', 'weather_condition',
                     'road_type', 'speed_limit', 'number_of_vehicles', 'number_of_casualties',
                     'severity', 'description']
    
    df_prep = df_prep[columns_needed]
    
    print(f"✅ Data prepared: {len(df_prep)} records ready for insertion")
    return df_prep


# ============================================================================
# 5. INSERT DATA INTO DATABASE (SIMPLE & FAST)
# ============================================================================
def insert_data_to_database(connection, df_prep):
    """Insert prepared data into accidents table"""
    cursor = connection.cursor()
    
    print("\n📥 Inserting data into database...")
    
    total = len(df_prep)
    batch_size = 10000  # Smaller batches = faster starts
    insert_count = 0
    error_count = 0
    
    insert_sql = """
        INSERT INTO accidents 
        (date, time, location, latitude, longitude, weather_condition, 
         road_type, speed_limit, number_of_vehicles, number_of_casualties, 
         severity, description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    print(f"📊 Batch size: {batch_size:,} | Total: {total:,} records\n")
    
    # Insert directly from DataFrame - NO conversion loop
    for i in range(0, len(df_prep), batch_size):
        batch_df = df_prep.iloc[i:i + batch_size]
        batch_data = [tuple(row) for row in batch_df.values]
        
        try:
            cursor.executemany(insert_sql, batch_data)
            connection.commit()
            insert_count += len(batch_data)
            
            percentage = (insert_count / total) * 100
            print(f"✅ [{percentage:6.2f}%] {insert_count:,} / {total:,} records")
            
        except Error as e:
            print(f"⚠️  Batch error: {str(e)[:80]}")
            connection.rollback()
            error_count += len(batch_data)
            continue
    
    cursor.close()
    print(f"\n✨ DONE! Inserted: {insert_count:,} records\n")


# ============================================================================
# 6. VERIFY DATA
# ============================================================================
def verify_data(connection):
    """Verify data was inserted correctly"""
    cursor = connection.cursor()
    
    print("\n" + "=" * 80)
    print("✅ DATA VERIFICATION")
    print("=" * 80)
    
    # Query 1: Total count
    cursor.execute("SELECT COUNT(*) FROM accidents")
    total = cursor.fetchone()[0]
    print(f"\n📊 Total Records: {total}")
    
    # Query 2: Severity distribution
    print(f"\n📊 Severity Distribution:")
    cursor.execute("""
        SELECT severity, COUNT(*) as count
        FROM accidents 
        GROUP BY severity
        ORDER BY count DESC
    """)
    for severity, count in cursor.fetchall():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   {severity}: {count} ({percentage:.1f}%)")
    
    # Query 3: Top locations
    print(f"\n📊 Top 5 Locations:")
    cursor.execute("""
        SELECT location, COUNT(*) as count
        FROM accidents 
        GROUP BY location
        ORDER BY count DESC
        LIMIT 5
    """)
    for location, count in cursor.fetchall():
        print(f"   {location}: {count} accidents")
    
    # Query 4: Sample data
    print(f"\n📊 Sample Records (First 3):")
    cursor.execute("""
        SELECT accident_id, date, time, location, severity, number_of_vehicles, number_of_casualties
        FROM accidents
        LIMIT 3
    """)
    for row in cursor.fetchall():
        print(f"   ID: {row[0]}, Date: {row[1]}, Location: {row[3]}, Severity: {row[4]}")

# ============================================================================
# 7. EXPORT TO CSV
# ============================================================================
def export_to_csv(connection):
    """Export database data to CSV"""
    print("\n" + "=" * 80)
    print("💾 EXPORTING DATA TO CSV")
    print("=" * 80)
    
    try:
        query = "SELECT * FROM accidents LIMIT 10000"
        df = pd.read_sql(query, connection)
        df.to_csv('accidents_export.csv', index=False)
        print(f"✅ Exported {len(df)} records to 'accidents_export.csv'")
        
    except Exception as e:
        print(f"❌ Error exporting: {e}")

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================
def main():
    connection = create_connection()
    
    if connection:
        try:
            # Load CSV data
            df = load_csv_data()
            
            if df is not None:
                # Prepare data
                df_prep = prepare_data(df)
                
                # Insert into database
                insert_data_to_database(connection, df_prep)
                
                # Verify insertion
                verify_data(connection)
                
                # Export to CSV
                export_to_csv(connection)
                
                print("\n" + "=" * 80)
                print("✅ IMPORT COMPLETE!")
                print("=" * 80)
                print("\n🎉 Your accident data is now in MySQL!")
                print("Next step: Run 'python complete_analysis.py' to analyze the data")
                
            else:
                print("\n❌ Could not load CSV data")
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            if connection.is_connected():
                connection.close()
                print("\n✅ Database connection closed")
    else:
        print("\n❌ Failed to connect to MySQL database")
        print("⚠️ Make sure:")
        print("   1. MySQL Server is running")
        print("   2. Database 'my_database' was created")
        print("   3. Password in db_config is correct")

if __name__ == "__main__":
    main()
