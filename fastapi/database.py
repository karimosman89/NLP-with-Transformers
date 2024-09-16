import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    designation TEXT,
    description TEXT,
    image_id INTEGER,
    prdtypecode INTEGER
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    description TEXT,
    image TEXT,
    prediction TEXT
);
""")

conn.commit()
conn.close()
