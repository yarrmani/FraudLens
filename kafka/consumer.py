import os
import sys
import json
import joblib
import numpy as np
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Consumer, KafkaError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
from feature_extractor import FeatureExtractor

# Load ML model
model = joblib.load(os.path.join(BASE_DIR, 'model', 'model_artifacts', 'fraud_model.pkl'))
extractor = FeatureExtractor()

# Kafka consumer config
consumer = Consumer({
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'group.id': 'fraudlens-scorer',
    'auto.offset.reset': 'latest'
})

RAW_TOPIC = os.getenv('KAFKA_RAW_TOPIC', 'raw-transactions')
consumer.subscribe([RAW_TOPIC])

# DB connection
def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        database=os.getenv("POSTGRES_DB", "frauddb"),
        user=os.getenv("POSTGRES_USER", "frauduser"),
        password=os.getenv("POSTGRES_PASSWORD", "fraudpass")
    )

def save_transaction(transaction: dict, fraud_score: float, is_flagged: bool, top_features: dict):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions
            (user_id, amount, hour, amount_scaled, fraud_score, is_flagged, top_features)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING transaction_id
    """, (
        transaction['user_id'],
        transaction['amount'],
        datetime.strptime(transaction['timestamp'], '%Y-%m-%d %H:%M:%S').hour,
        float(extractor.extract(transaction)[0][-1]),
        round(fraud_score, 4),
        is_flagged,
        json.dumps(top_features)
    ))
    transaction_id = str(cursor.fetchone()[0])
    conn.commit()
    cursor.close()
    conn.close()
    return transaction_id

print(f"[Consumer] Listening on topic: {RAW_TOPIC}")
print("[Consumer] Press Ctrl+C to stop\n")

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            print(f"[Consumer] Error: {msg.error()}")
            continue

        # Parse transaction
        transaction = json.loads(msg.value().decode('utf-8'))

        # Score with ML model
        feature_array = extractor.extract(transaction)
        fraud_score = float(model.predict_proba(feature_array)[0][1])
        is_flagged = fraud_score >= 0.5

        # Get top features
        feature_importances = model.feature_importances_
        feature_names = extractor.feature_columns
        top_indices = np.argsort(feature_importances)[-3:][::-1]
        top_features = {
            feature_names[i]: round(float(feature_importances[i]), 4)
            for i in top_indices
        }

        # Save to PostgreSQL
        transaction_id = save_transaction(transaction, fraud_score, is_flagged, top_features)

        status = "🚨 FRAUD" if is_flagged else "✅ clean"
        print(f"[Consumer] {status} | user={transaction['user_id']} "
              f"amount=${transaction['amount']} score={fraud_score:.4f} "
              f"id={transaction_id[:8]}...")

except KeyboardInterrupt:
    print("\n[Consumer] Stopping...")
finally:
    consumer.close()
    extractor.close()
    print("[Consumer] Done.")