CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    user_id                 SERIAL PRIMARY KEY,
    name                    VARCHAR(100) NOT NULL,
    email                   VARCHAR(150) UNIQUE NOT NULL,
    home_country            VARCHAR(100),
    avg_transaction_amount  DECIMAL(10, 2) DEFAULT 0.00,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          INTEGER REFERENCES users(user_id),
    amount           DECIMAL(10, 2) NOT NULL,
    hour             INTEGER CHECK (hour >= 0 AND hour <= 23),
    amount_scaled    DECIMAL(10, 4),
    fraud_score      DECIMAL(5, 4),
    is_flagged       BOOLEAN DEFAULT FALSE,
    top_features     JSONB,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fraud_reviews (
    review_id          SERIAL PRIMARY KEY,
    transaction_id     UUID REFERENCES transactions(transaction_id),
    officer_decision   VARCHAR(20) CHECK (
                           officer_decision IN ('confirmed_fraud', 'false_positive')
                       ),
    notes              TEXT,
    reviewed_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_flagged ON transactions(is_flagged);
CREATE INDEX IF NOT EXISTS idx_transactions_created ON transactions(created_at DESC);
