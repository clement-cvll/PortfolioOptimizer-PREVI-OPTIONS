import psycopg2
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Connect to the database
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="password"
)

# Create a cursor object
cur = conn.cursor()

# Drop the table if it exists
cur.execute("DROP TABLE IF EXISTS opcvm_data;")

# Create a hypertable if it doesn't exist
cur.execute("""
    CREATE TABLE opcvm_data ( 
         date DATE, 
         open FLOAT, 
         high FLOAT, 
         low FLOAT, 
         close FLOAT,  
         ticker TEXT,
         name TEXT
    )
""")

cur.execute("""
    SELECT create_hypertable('opcvm_data', 'date', if_not_exists => TRUE);
""")
# Commit the changes to the database
conn.commit()

# Insert data into the hypertable
df = pd.read_csv("tickers.csv", index_col=0)
for ticker in tqdm(df.index, desc="Uploading data to the database"):
    data = yf.Ticker(ticker).history(period="max")
    data.reset_index(inplace=True)
    
    # Iterate through each row of the DataFrame
    for _, row in data.iterrows():
        cur.execute("""
            INSERT INTO opcvm_data (date, open, high, low, close, ticker, name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            row["Date"],
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            ticker,
            df.loc[ticker, "name"]
        ))

conn.commit()
cur.close()
conn.close()