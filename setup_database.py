import mysql.connector
from mysql.connector import Error

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Gskohli2804$',  # Change if needed
    'database': 'my_database'
}

def create_database_and_tables():
    """Create database and all required tables"""
    try:
        # First connection (without database)
        print("🔗 Connecting to MySQL Server...")
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        if connection.is_connected():
            print("✅ Connected to MySQL Server!")
        
        cursor = connection.cursor()
        
        # Create Database
        print("\n📊 Creating database 'my_database'...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS my_database")
        print("✅ Database created/verified")
        
        # Select database
        cursor.execute("USE my_database")
        
        # Create accidents table
        print("\n📝 Creating 'accidents' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accidents (
                ID INT AUTO_INCREMENT PRIMARY KEY,
                Severity INT,
                Start_Time DATETIME,
                End_Time DATETIME,
                Start_Lat FLOAT,
                Start_Lng FLOAT,
                Distance_mi FLOAT,
                Description TEXT,
                Street VARCHAR(255),
                City VARCHAR(100),
                State VARCHAR(50),
                Temperature_F FLOAT,
                Humidity FLOAT,
                Visibility_mi FLOAT,
                Wind_Speed_mph FLOAT,
                Weather_Condition VARCHAR(255),
                Amenity BOOLEAN,
                Bump BOOLEAN,
                Crossing BOOLEAN,
                Give_Way BOOLEAN,
                Junction BOOLEAN,
                No_Exit BOOLEAN,
                Railway BOOLEAN,
                Roundabout BOOLEAN,
                Station BOOLEAN,
                Stop BOOLEAN,
                Traffic_Calming BOOLEAN,
                Traffic_Signal BOOLEAN,
                Turning_Loop BOOLEAN
            )
        """)
        print("✅ 'accidents' table created/verified")
        
        # Create severity_levels table
        print("\n📝 Creating 'severity_levels' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS severity_levels (
                severity_id INT PRIMARY KEY,
                description VARCHAR(255)
            )
        """)
        print("✅ 'severity_levels' table created/verified")
        
        # Insert severity levels
        print("\n📝 Inserting severity level data...")
        cursor.execute("INSERT IGNORE INTO severity_levels VALUES (1, 'Low')")
        cursor.execute("INSERT IGNORE INTO severity_levels VALUES (2, 'Medium')")
        cursor.execute("INSERT IGNORE INTO severity_levels VALUES (3, 'High')")
        cursor.execute("INSERT IGNORE INTO severity_levels VALUES (4, 'Critical')")
        
        connection.commit()
        print("✅ Severity levels inserted")
        
        print("\n" + "=" * 80)
        print("✅ DATABASE SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        connection.close()
        return True
        
    except Error as e:
        print(f"\n❌ Error: {e}")
        print("⚠️ Make sure:")
        print("   1. MySQL Server is running")
        print("   2. Password is correct")
        print("   3. User 'root' exists")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🗄️  SETUP DATABASE FOR ACCIDENT SEVERITY PROJECT")
    print("=" * 80)
    
    success = create_database_and_tables()
    
    if success:
        print("\n✅ Next step: python import_accident_data.py")
    else:
        print("\n❌ Setup failed!")
    
    print("=" * 80)
