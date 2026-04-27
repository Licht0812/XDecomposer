import sqlite3
import os
import json
import numpy as np

db_path = "/data/group/project1/Crystal/UniqCryLabeled.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_name = cursor.fetchone()[0]

cursor.execute(f"SELECT numbers, positions, cell FROM {table_name} LIMIT 1")
row = cursor.fetchone()

print(f"Numbers type: {type(row[0])}")
print(f"Positions type: {type(row[1])}")
print(f"Cell type: {type(row[2])}")

# Try to decode if they are blobs
def decode(data):
    try:
        return np.frombuffer(data)
    except:
        return data

print(f"Numbers decoded: {decode(row[0])}")
print(f"Positions decoded shape: {decode(row[1]).shape if hasattr(decode(row[1]), 'shape') else 'N/A'}")
print(f"Cell decoded: {decode(row[2])}")

conn.close()
