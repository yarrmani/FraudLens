import sys
import os
import numpy as np
import random
from datetime import datetime

# Add backend/ to path so we can import feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import FeatureExtractor

print("=" * 50)
print("FraudLens — Feature Extractor Test")
print("=" * 50)

# Initialize
extractor = FeatureExtractor()

# Build a fake transaction with random V1-V28 values
fake_transaction = {
    "user_id": 1,
    "amount": 150.0,
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    **{f"V{i}": round(random.uniform(-3.0, 3.0), 6) for i in range(1, 29)}
}

print(f"\nFake transaction amount: ${fake_transaction['amount']}")
print(f"Timestamp: {fake_transaction['timestamp']}")

# Test extract()
print("\n--- Testing extract() ---")
feature_array = extractor.extract(fake_transaction)
print(f"Output shape:   {feature_array.shape}")
print(f"Hour extracted: {int(feature_array[0][-2])}")
print(f"Amount scaled:  {feature_array[0][-1]:.6f}")

# Confirm shape is exactly (1, 30)
assert feature_array.shape == (1, 30), f"SHAPE ERROR: got {feature_array.shape}"
print("Shape assertion passed: (1, 30) ✓")

# Test get_user_context()
print("\n--- Testing get_user_context(user_id=1) ---")
context = extractor.get_user_context(user_id=1)
print(f"Average amount:        ${context['avg_amount']}")
print(f"Home country:          {context['home_country']}")
print(f"Transactions last 24h: {context['transactions_last_24h']}")
print(f"Recent amounts:        {context['recent_amounts']}")

# Close connection
extractor.close()

print("\n" + "=" * 50)
print("All tests passed. Feature extractor is working.")
print("=" * 50)