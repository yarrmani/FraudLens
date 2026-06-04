import os
import joblib
import numpy as np
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Build absolute path to project root (one level up from backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
load_dotenv(os.path.join(BASE_DIR, '.env'))


class FeatureExtractor:
    """
    Bridges incoming transactions with the ML model.
    Handles feature engineering and database context queries.
    """

    def __init__(self):
        # Load saved ML artifacts using absolute paths
        self.scaler = joblib.load(
            os.path.join(BASE_DIR, 'model', 'model_artifacts', 'scaler.pkl')
        )
        self.feature_columns = joblib.load(
            os.path.join(BASE_DIR, 'model', 'model_artifacts', 'feature_columns.pkl')
        )

        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5433)),
            database=os.getenv("POSTGRES_DB", "frauddb"),
            user=os.getenv("POSTGRES_USER", "frauduser"),
            password=os.getenv("POSTGRES_PASSWORD", "fraudpass")
        )
        print("[FeatureExtractor] Initialized. DB connected, artifacts loaded.")

    def extract(self, transaction: dict) -> np.ndarray:
        """
        Takes a raw transaction dict and returns a (1, 30) numpy array
        ready to be passed directly into model.predict_proba()

        Expected transaction keys:
        - user_id: int
        - amount: float
        - timestamp: str  e.g. "2025-01-15 14:32:00"
        - V1 through V28: float (PCA features)
        """

        # Step 1 — Extract hour from timestamp
        ts = datetime.strptime(transaction['timestamp'], '%Y-%m-%d %H:%M:%S')
        hour = ts.hour

        # Step 2 — Scale the amount using the saved RobustScaler
        # Must reshape to (1,1) because scaler expects a 2D array
        amount_scaled = self.scaler.transform(
            np.array([[transaction['amount']]])
        )[0][0]

        # Step 3 — Build feature dict in the exact order the model expects
        feature_dict = {}

        # Add V1-V28 directly from transaction
        for i in range(1, 29):
            key = f'V{i}'
            feature_dict[key] = transaction.get(key, 0.0)

        # Add engineered features
        feature_dict['Hour'] = hour
        feature_dict['Amount_Scaled'] = amount_scaled

        # Step 4 — Convert to ordered numpy array matching feature_columns
        feature_array = np.array(
            [feature_dict[col] for col in self.feature_columns]
        ).reshape(1, -1)

        return feature_array

    def get_user_context(self, user_id: int) -> dict:
        """
        Queries PostgreSQL for behavioral context about a user.
        Used for dashboard display and anomaly explanation.
        Returns avg spend, home country, recent activity.
        """
        cursor = self.conn.cursor()

        try:
            # Query user profile
            cursor.execute("""
                SELECT avg_transaction_amount, home_country
                FROM users
                WHERE user_id = %s
            """, (user_id,))

            user_row = cursor.fetchone()

            if user_row is None:
                return {
                    "avg_amount": 0.0,
                    "home_country": "Unknown",
                    "transactions_last_24h": 0,
                    "recent_amounts": []
                }

            avg_amount = float(user_row[0])
            home_country = user_row[1]

            # Count transactions in the last 24 hours
            cursor.execute("""
                SELECT COUNT(*)
                FROM transactions
                WHERE user_id = %s
                AND created_at >= NOW() - INTERVAL '24 hours'
            """, (user_id,))

            txn_count = cursor.fetchone()[0]

            # Get last 10 transaction amounts for context
            cursor.execute("""
                SELECT amount
                FROM transactions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (user_id,))

            recent_amounts = [float(row[0]) for row in cursor.fetchall()]

            return {
                "avg_amount": avg_amount,
                "home_country": home_country,
                "transactions_last_24h": int(txn_count),
                "recent_amounts": recent_amounts
            }

        except Exception as e:
            print(f"[FeatureExtractor] Error querying user context: {e}")
            return {
                "avg_amount": 0.0,
                "home_country": "Unknown",
                "transactions_last_24h": 0,
                "recent_amounts": []
            }

        finally:
            cursor.close()

    def close(self):
        """Close the database connection cleanly."""
        if self.conn:
            self.conn.close()
            print("[FeatureExtractor] Database connection closed.")