import psycopg2
import random
import os
from faker import Faker
from dotenv import load_dotenv

load_dotenv()
fake = Faker()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "frauddb"),
    user=os.getenv("POSTGRES_USER", "frauduser"),
    password=os.getenv("POSTGRES_PASSWORD", "fraudpass")
)

cursor = conn.cursor()
print("Seeding 50 users...")

for _ in range(50):
    cursor.execute("""
        INSERT INTO users (name, email, home_country, avg_transaction_amount)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (email) DO NOTHING
    """, (
        fake.name(),
        fake.unique.email(),
        random.choice(['UK', 'UAE', 'Poland', 'Portugal', 'Germany']),
        round(random.uniform(20, 500), 2)
    ))

conn.commit()
cursor.close()
conn.close()
print("Done. 50 users seeded.")
