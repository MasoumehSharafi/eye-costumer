BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 1. Governed knowledge sources
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_sources (
    registry_id TEXT PRIMARY KEY,

    source_name TEXT NOT NULL,
    content_type TEXT NOT NULL,

    language VARCHAR(2) NOT NULL
        CHECK (language IN ('en', 'fr')),

    country TEXT NOT NULL DEFAULT 'CA',

    audience TEXT NOT NULL
        CHECK (
            audience IN (
                'internal',
                'doctor',
                'customer',
                'shared'
            )
        ),

    approval_status TEXT NOT NULL
        CHECK (
            approval_status IN (
                'draft',
                'approved',
                'expired',
                'rejected'
            )
        ),

    approver TEXT,
    knowledge_version TEXT NOT NULL,

    review_date DATE,
    expires_at DATE,

    notes TEXT,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- 2. Structured products
-- ============================================================

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT NOT NULL,
    language VARCHAR(2) NOT NULL
        CHECK (language IN ('en', 'fr')),

    product_handle TEXT NOT NULL,
    title TEXT,

    country TEXT NOT NULL DEFAULT 'CA',

    source_registry_id TEXT NOT NULL
        REFERENCES knowledge_sources(registry_id)
        ON DELETE RESTRICT,

    url TEXT,

    PRIMARY KEY (
        product_id,
        language
    )
);


CREATE UNIQUE INDEX IF NOT EXISTS
    idx_products_handle_language
ON products (
    product_handle,
    language
);


CREATE INDEX IF NOT EXISTS
    idx_products_source_registry
ON products (
    source_registry_id
);


-- ============================================================
-- 3. Searchable knowledge passages
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_passages (
    passage_id TEXT PRIMARY KEY,

    source_registry_id TEXT NOT NULL
        REFERENCES knowledge_sources(registry_id)
        ON DELETE RESTRICT,

    product_id TEXT,
    product_language VARCHAR(2),

    section TEXT,

    content TEXT NOT NULL,

    metadata JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    embedding VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT passage_product_pair
        CHECK (
            (
                product_id IS NULL
                AND product_language IS NULL
            )
            OR
            (
                product_id IS NOT NULL
                AND product_language IS NOT NULL
            )
        ),

    CONSTRAINT passage_product_fk
        FOREIGN KEY (
            product_id,
            product_language
        )
        REFERENCES products (
            product_id,
            language
        )
        ON DELETE RESTRICT
);


CREATE INDEX IF NOT EXISTS
    idx_passages_source_registry
ON knowledge_passages (
    source_registry_id
);


CREATE INDEX IF NOT EXISTS
    idx_passages_product
ON knowledge_passages (
    product_id,
    product_language
);


CREATE INDEX IF NOT EXISTS
    idx_passages_section
ON knowledge_passages (
    section
);


CREATE INDEX IF NOT EXISTS
    idx_passages_metadata
ON knowledge_passages
USING GIN (
    metadata
);


COMMIT;
