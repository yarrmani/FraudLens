import os
import sys
import json
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Kafka config
producer = Producer({
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
})

TOPIC = os.getenv('KAFKA_RAW_TOPIC', 'raw-transactions')

def generate_transaction(user_id: int, fraudulent: bool = False) -> dict:
    """Generate a fake transaction. Fraudulent ones have extreme V values."""
    if fraudulent:
        v_values = {f"V{i}": round(random.uniform(-5.0, 5.0), 6) for i in range(1, 29)}
        amount = round(random.uniform(500, 3000), 2)
    else:
        v_values = {f"V{i}": round(random.uniform(-1.0, 1.0), 6) for i in range(1, 29)}
        amount = round(random.uniform(5, 200), 2)

    return {
        "user_id": user_id,
        "amount": amount,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **v_values
    }

def delivery_report(err, msg):
    if err:
        print(f"[Producer] Delivery failed: {err}")
    else:
        print(f"[Producer] Sent to {msg.topic()} partition {msg.partition()}")

print(f"[Producer] Starting. Sending to topic: {TOPIC}")
print("[Producer] Press Ctrl+C to stop\n")

count = 0
try:
    while True:
        # Every 10th transaction is fraudulent
        is_fraud = (count % 10 == 0)
        user_id = random.randint(1, 50)

        transaction = generate_transaction(user_id, fraudulent=is_fraud)

        producer.produce(
            TOPIC,
            key=str(user_id),
            value=json.dumps(transaction),
            callback=delivery_report
        )
        producer.poll(0)

        label = "FRAUD " if is_fraud else "normal"
        print(f"[Producer] #{count+1} [{label}] user={user_id} amount=${transaction['amount']}")

        count += 1
        time.sleep(2)  # one transaction every 2 seconds

except KeyboardInterrupt:
    print("\n[Producer] Stopping...")
    producer.flush()
    print("[Producer] Done.")