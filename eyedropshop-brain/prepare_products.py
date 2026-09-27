from pathlib import Path
import html
import json
import re

import pandas as pd


EN_FILE = "knowledge/products/Active Products CA EN.xlsx"
FR_FILE = "knowledge/products/Active Products CA FR.xlsx"
OUTPUT_FOLDER = Path("prepared_data")

OUTPUT_FOLDER.mkdir(exist_ok=True)


FIELD_MAPPING = {
    "ml_fr.title": "title",
    "ml_fr.description": "description",
    "custom.accordion_benefits": "benefits",
    "custom.more_details": "more_details",
    "custom.how_to_use": "how_to_use",
    "custom.ingredients": "ingredients",
    "custom.warnings": "warnings",
    "custom.dosage": "dosage",
    "custom.product_details": "product_details",
    "custom.features": "features",
}


ENGLISH_COLUMNS = {
    "title": "Title",
    "description": "Body HTML",
    "benefits": (
        "Metafield: custom.accordion_benefits "
        "[rich_text_field]"
    ),
    "more_details": (
        "Metafield: custom.more_details "
        "[rich_text_field]"
    ),
    "how_to_use": (
        "Metafield: custom.how_to_use "
        "[rich_text_field]"
    ),
    "ingredients": (
        "Metafield: custom.ingredients "
        "[rich_text_field]"
    ),
    "warnings": (
        "Metafield: custom.warnings "
        "[rich_text_field]"
    ),
    "dosage": (
        "Metafield: custom.dosage "
        "[rich_text_field]"
    ),
    "product_details": (
        "Metafield: custom.product_details "
        "[rich_text_field]"
    ),
    "features": (
        "Metafield: custom.features "
        "[rich_text_field]"
    ),
}


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    # Extract text from Shopify rich-text JSON.
    try:
        parsed = json.loads(text)

        def extract_json_text(item):
            if isinstance(item, dict):
                parts = []

                if isinstance(item.get("value"), str):
                    parts.append(item["value"])

                for child in item.get("children", []):
                    parts.extend(extract_json_text(child))

                return parts

            if isinstance(item, list):
                parts = []

                for child in item:
                    parts.extend(extract_json_text(child))

                return parts

            return []

        extracted = extract_json_text(parsed)

        if extracted:
            text = " ".join(extracted)

    except (json.JSONDecodeError, TypeError):
        pass

    # Remove HTML.
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------- English products ----------

english = pd.read_excel(
    EN_FILE,
    sheet_name="Products",
)

# Keep only one main row per product.
english = english[english["Top Row"].eq(1)].copy()

records = []

for _, row in english.iterrows():
    record = {
        "product_handle": clean_text(row.get("Handle")),
        "product_id": clean_text(row.get("ID")),
        "language": "en",
        "country": "CA",
        "source": Path(EN_FILE).name,
        "vendor": clean_text(row.get("Vendor")),
        "product_type": clean_text(row.get("Type")),
        "url": clean_text(row.get("URL")),
    }

    for output_field, source_column in ENGLISH_COLUMNS.items():
        record[output_field] = clean_text(
            row.get(source_column)
        )

    records.append(record)


# ---------- French translations ----------

french = pd.read_excel(
    FR_FILE,
    sheet_name="Translations",
)

# Keep current product translations only.
french = french[
    french["Parent Type"].eq("PRODUCT")
    & french["Handle"].isin(FIELD_MAPPING)
    & french["Translated content"].notna()
    & ~french["Is Outdated"].eq(True)
].copy()

french["clean_translation"] = french[
    "Translated content"
].apply(clean_text)

french = french[
    french["clean_translation"].ne("")
]

english_by_handle = {
    record["product_handle"]: record
    for record in records
    if record["language"] == "en"
}

for parent_handle, group in french.groupby("Parent Handle"):
    parent_handle = clean_text(parent_handle)
    english_record = english_by_handle.get(parent_handle)

    if english_record is None:
        continue

    french_record = {
        "product_handle": parent_handle,
        "product_id": english_record["product_id"],
        "language": "fr",
        "country": "CA",
        "source": Path(FR_FILE).name,
        "vendor": english_record["vendor"],
        "product_type": english_record["product_type"],
        "url": english_record["url"],
    }

    for field in ENGLISH_COLUMNS:
        french_record[field] = ""

    for _, translation in group.iterrows():
        output_field = FIELD_MAPPING[translation["Handle"]]
        french_record[output_field] = translation[
            "clean_translation"
        ]

    records.append(french_record)


products = pd.DataFrame(records)

products.to_csv(
    OUTPUT_FOLDER / "products_bilingual.csv",
    index=False,
    encoding="utf-8-sig",
)


# ---------- Translation coverage ----------

content_fields = list(ENGLISH_COLUMNS.keys())

coverage = (
    products.groupby("language")[content_fields]
    .agg(lambda column: column.fillna("").ne("").sum())
    .transpose()
    .reset_index()
    .rename(columns={"index": "field"})
)

coverage.to_csv(
    OUTPUT_FOLDER / "translation_coverage.csv",
    index=False,
    encoding="utf-8-sig",
)

print("Product preparation completed.")
print(f"Total records: {len(products)}")
print(products["language"].value_counts())

english_handles = set(
    products.loc[
        products["language"].eq("en"),
        "product_handle",
    ]
)

french_handles = set(
    products.loc[
        products["language"].eq("fr"),
        "product_handle",
    ]
)

print("\nEnglish products:", len(english_handles))
print("French products:", len(french_handles))
print("Products available in both:", len(english_handles & french_handles))
print("English products missing French:", len(english_handles - french_handles))
print("French products missing English:", len(french_handles - english_handles))
print("Are the product sets identical?", english_handles == french_handles)


fields_to_show = [
    "product_handle",
    "product_id",
    "language",
    "title",
    "description",
    "benefits",
    "more_details",
    "how_to_use",
    "ingredients",
    "warnings",
    "dosage",
    "product_details",
    "features",
    "vendor",
    "product_type",
    "url",
]

# Find a product available in both languages.
# Prefer one with the largest amount of bilingual information.
content_fields = [
    "title",
    "description",
    "benefits",
    "more_details",
    "how_to_use",
    "ingredients",
    "warnings",
    "dosage",
    "product_details",
    "features",
]

french_products = products[products["language"].eq("fr")].copy()

french_products["available_fields"] = (
    french_products[content_fields]
    .fillna("")
    .ne("")
    .sum(axis=1)
)

example_handle = (
    french_products
    .sort_values("available_fields", ascending=False)
    .iloc[0]["product_handle"]
)

example = products[
    products["product_handle"].eq(example_handle)
][fields_to_show]

example.to_csv(
    OUTPUT_FOLDER / "example_bilingual_product.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "Example saved to "
    "prepared_data/example_bilingual_product.csv"
)

required_fields = [
    "title",
    "description",
    "benefits",
    "more_details",
    "how_to_use",
    "ingredients",
    "warnings",
    "dosage",
    "product_details",
    "features",
]

english_records = (
    products[products["language"].eq("en")]
    .set_index("product_handle")
)

french_records = (
    products[products["language"].eq("fr")]
    .set_index("product_handle")
)

shared_handles = english_records.index.intersection(
    french_records.index
)

complete_products = []

for handle in shared_handles:
    english_complete = all(
        str(english_records.loc[handle, field]).strip()
        for field in required_fields
    )

    french_complete = all(
        str(french_records.loc[handle, field]).strip()
        for field in required_fields
    )

    if english_complete and french_complete:
        complete_products.append(handle)

print(
    "Products with all fields in both languages:",
    len(complete_products),
)

pd.DataFrame({
    "product_handle": complete_products
}).to_csv(
    OUTPUT_FOLDER / "complete_bilingual_products.csv",
    index=False,
    encoding="utf-8-sig",
)

REQUIRED_PROTOTYPE_FIELDS = [
    "how_to_use",
    "ingredients",
    "warnings",
]

PROTOTYPE_PRODUCT_COUNT = 30


english_records = (
    products[products["language"].eq("en")]
    .set_index("product_handle")
)

french_records = (
    products[products["language"].eq("fr")]
    .set_index("product_handle")
)

shared_handles = english_records.index.intersection(
    french_records.index
)

eligible_handles = []

for handle in shared_handles:
    english_complete = all(
        str(english_records.loc[handle, field]).strip()
        for field in REQUIRED_PROTOTYPE_FIELDS
    )

    french_complete = all(
        str(french_records.loc[handle, field]).strip()
        for field in REQUIRED_PROTOTYPE_FIELDS
    )

    if english_complete and french_complete:
        eligible_handles.append(handle)


# Choose the first 30 deterministically.
selected_handles = sorted(
    eligible_handles
)[:PROTOTYPE_PRODUCT_COUNT]

prototype_products = products[
    products["product_handle"].isin(selected_handles)
].copy()

prototype_products = prototype_products.sort_values(
    ["product_handle", "language"]
)

prototype_products.to_csv(
    OUTPUT_FOLDER / "prototype_30_bilingual_products.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame({
    "product_handle": selected_handles
}).to_csv(
    OUTPUT_FOLDER / "prototype_30_product_list.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nEligible bilingual products:", len(eligible_handles))
print("Selected products:", len(selected_handles))
print("Saved bilingual records:", len(prototype_products))