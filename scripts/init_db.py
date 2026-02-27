"""Create sample table in Postgres for SQL tool demos. Run once after Postgres is up."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings
from sqlalchemy import create_engine, text

def main():
    s = get_settings()
    engine = create_engine(s.postgres_dsn)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200),
                price DECIMAL(10,2)
            )
        """))
        conn.execute(text("DELETE FROM products"))
        conn.execute(text("""
            INSERT INTO products (name, price) VALUES
            ('Widget A', 10.50),
            ('Widget B', 25.00),
            ('Gadget X', 99.99)
        """))
        conn.commit()
    print("DB initialized: table products with sample rows.")

if __name__ == "__main__":
    main()
