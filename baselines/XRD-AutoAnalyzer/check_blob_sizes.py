
import sqlite3
import numpy as np

def check_blob_sizes(db_path, cid):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT numbers, positions, cell FROM systems WHERE id=?", (cid,))
    row = cur.fetchone()
    conn.close()

    if row:
        print(f"ID {cid} Blob sizes: numbers={len(row[0])}, positions={len(row[1])}, cell={len(row[2])}")

db_path = 'data/UniqCryLabeled.db'
cid = 8604
check_blob_sizes(db_path, cid)
