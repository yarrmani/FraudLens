import os
import sys
import json
import random
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

CSV_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"
)

producer = Producer({
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
})
TOPIC = os.getenv('KAFKA_RAW_TOPIC', 'raw-transactions')

print("[Replay] Loading real dataset...")
df = pd.read_csv(CSV_PATH)

fraud_rows = df[df['Class'] == 1].reset_index(drop=True)
normal_rows = df[df['Class'] == 0].reset_index(drop=True)

print(f"[Replay] Loaded {len(fraud_rows)} real fraud rows, {len(normal_rows)} real normal rows")

V_COLUMNS = [f'V{i}' for i in range(1, 29)]


def row_to_transaction(row, user_id):
    transaction = {
        "user_id": user_id,
        "amount": float(row['Amount']),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    for col in V_COLUMNS:
        transaction[col] = float(row[col])
    return transaction


def delivery_report(err, msg):
    if err:
        print(f"[Replay] Delivery failed: {err}")


print(f"[Replay] Starting. Sending REAL fraud + normal rows to topic: {TOPIC}")
print("[Replay] Roughly 1 in 10 transactions will be REAL fraud")
print("[Replay] Press Ctrl+C to stop\n")

count = 0
try:
    while True:
        is_fraud = (count % 10 == 0)
        user_id = random.randint(1, 50)

        if is_fraud:
            row = fraud_rows.sample(1).iloc[0]
            label = "FRAUD "
        else:
            row = normal_rows.sample(1).iloc[0]
            label = "normal"

        transaction = row_to_transaction(row, user_id)

        producer.produce(
            TOPIC,
            key=str(user_id),
            value=json.dumps(transaction),
            callback=delivery_report
        )
        producer.poll(0)

        print(f"[Replay] #{count+1} [{label}] user={user_id} "
              f"amount=${transaction['amount']:.2f} (real {label.strip().lower()} row)")

        count += 1
        time.sleep(2)

except KeyboardInterrupt:
    print("\n[Replay] Stopping...")
    producer.flush()
    print("[Replay] Done.")
