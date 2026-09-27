from pathlib import Path

from governance_registry import (
    get_governance_record,
    governance_metadata,
    record_is_indexable,
)

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from ingestion_validator import validate_products


PRODUCT_FILE = Path(
    "prototype_data/products/products_30_bilingual.csv"
)
DATABASE_FOLDER = "chroma_products_db"
COLLECTION_NAME = "eyedropshop_products"
EMBEDDING_MODEL = "BAAI/bge-m3"


SECTION_GROUPS = {
    "description": [
        "title",
        "description",
    ],
    "benefits_and_details": [
        "benefits",
        "more_details",
        "product_details",
        "features",
    ],
    "usage": [
        "how_to_use",
        "dosage",
    ],
    "ingredients": [
        "ingredients",
    ],
    "warnings": [
        "warnings",
    ],
}


SECTION_LABELS = {
    "en": {
        "title": "Product",
        "description": "Description",
        "benefits": "Benefits",
        "more_details": "More details",
        "product_details": "Product details",
        "features": "Features",
        "how_to_use": "How to use",
        "dosage": "Dosage",
        "ingredients": "Ingredients",
        "warnings": "Warnings",
    },
    "fr": {
        "title": "Produit",
        "description": "Description",
        "benefits": "Avantages",
        "more_details": "Plus de détails",
        "product_details": "Détails du produit",
        "features": "Caractéristiques",
        "how_to_use": "Mode d’emploi",
        "dosage": "Dosage",
        "ingredients": "Ingrédients",
        "warnings": "Avertissements",
    },
}


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()

print("Running product ingestion validation...")

if not validate_products():
    raise SystemExit(
        "Product ingestion validation failed. "
        "Database build stopped."
    )

products = pd.read_csv(PRODUCT_FILE)

documents = []
document_ids = []
metadatas = []

for row_number, row in products.iterrows():
    source = clean_value(row["source"])
    language = clean_value(row["language"])

    governance_record = get_governance_record(
        source=source,
        content_type="product",
        language=language,
        country="CA",
    )

    if not record_is_indexable(
        governance_record
    ):
        print(
            "[governance] skipping "
            f"non-indexable source: {source}"
        )
        continue

    source_governance = governance_metadata(
        governance_record
    )

    labels = SECTION_LABELS[language]

    product_handle = clean_value(
        row["product_handle"]
    )
    product_id = clean_value(
        row["product_id"]
    )
    product_title = clean_value(
        row["title"]
    )

    for section_name, fields in SECTION_GROUPS.items():
        section_parts = []

        for field in fields:
            value = clean_value(row.get(field))

            if value:
                section_parts.append(
                    f"{labels[field]}: {value}"
                )

        if not section_parts:
            continue

        document = "\n".join(section_parts)

        documents.append(document)

        document_ids.append(
            f"{product_id}-{language}-{section_name}"
        )

        metadata = {
            "content_type": "product",
            "product_id": product_id,
            "product_handle": product_handle,
            "product_title": product_title,
            "language": language,
            "country": "CA",
            "section": section_name,
            "source": source,
            "url": clean_value(row["url"]),
        }

        metadata.update(
            source_governance
        )

        metadatas.append(
            metadata
        )

print("Loading BGE-M3...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Creating embeddings for {len(documents)} passages...")

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True,
).tolist()

client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME
)

collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)

print("Product database created successfully.")
print("Products:", products["product_handle"].nunique())
print("English records:", products["language"].eq("en").sum())
print("French records:", products["language"].eq("fr").sum())
print("Searchable passages:", collection.count())