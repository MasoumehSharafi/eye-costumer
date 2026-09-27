from pathlib import Path

import pandas as pd


EN_FILE = "knowledge/products/Active Products CA EN.xlsx"
FR_FILE = "knowledge/products/Active Products CA FR.xlsx"
OUTPUT_FOLDER = Path("inspection_results")

OUTPUT_FOLDER.mkdir(exist_ok=True)

english = pd.read_excel(EN_FILE, sheet_name="Products")
french = pd.read_excel(FR_FILE, sheet_name="Translations")


# Save the general summary
summary = [
    {"section": "English", "metric": "Total rows", "value": len(english)},
    {
        "section": "English",
        "metric": "Unique product IDs",
        "value": english["ID"].nunique(),
    },
    {
        "section": "English",
        "metric": "Unique handles",
        "value": english["Handle"].nunique(),
    },
    {
        "section": "English",
        "metric": "Top rows",
        "value": english["Top Row"].eq(1).sum(),
    },
    {"section": "French", "metric": "Total rows", "value": len(french)},
    {
        "section": "French",
        "metric": "Translated content available",
        "value": french["Translated content"].notna().sum(),
    },
    {
        "section": "French",
        "metric": "Outdated translations",
        "value": french["Is Outdated"].eq(True).sum(),
    },
    {
        "section": "French",
        "metric": "Parent handles available",
        "value": french["Parent Handle"].notna().sum(),
    },
]

pd.DataFrame(summary).to_csv(
    OUTPUT_FOLDER / "inspection_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# Save all French translation records
columns = [
    "ID",
    "Identification",
    "Handle",
    "Parent Type",
    "Parent ID",
    "Parent Handle",
    "Field",
    "Locale",
    "Market",
    "Default content",
    "Translated content",
    "Is Outdated",
    "Updated At",
]

french[columns].to_csv(
    OUTPUT_FOLDER / "french_translation_records.csv",
    index=False,
    encoding="utf-8-sig",
)

# Count the French translation fields
identification_counts = (
    french.groupby(
        ["Identification", "Parent Type"],
        dropna=False,
    )
    .agg(
        total_records=("ID", "size"),
        translated_records=(
            "Translated content",
            lambda values: values.notna().sum(),
        ),
        outdated_records=(
            "Is Outdated",
            lambda values: values.eq(True).sum(),
        ),
    )
    .reset_index()
)

identification_counts.to_csv(
    OUTPUT_FOLDER / "french_field_counts.csv",
    index=False,
    encoding="utf-8-sig",
)


# Save a small sample of translated records
translation_samples = (
    french.dropna(subset=["Translated content"])
    [
        [
            "Identification",
            "Parent Type",
            "Parent ID",
            "Parent Handle",
            "Default content",
            "Translated content",
            "Is Outdated",
        ]
    ]
    .head(50)
)

translation_samples.to_csv(
    OUTPUT_FOLDER / "french_translation_samples.csv",
    index=False,
    encoding="utf-8-sig",
)

# Create the correct summary using the metafield Handle
field_summary = (
    french.groupby(
        ["Handle", "Parent Type"],
        dropna=False,
    )
    .agg(
        total_records=("ID", "size"),
        translated_records=(
            "Translated content",
            lambda values: values.notna().sum(),
        ),
        outdated_records=(
            "Is Outdated",
            lambda values: values.eq(True).sum(),
        ),
    )
    .reset_index()
    .sort_values(
        by="translated_records",
        ascending=False,
    )
)

field_summary.to_csv(
    OUTPUT_FOLDER / "french_metafield_summary.csv",
    index=False,
    encoding="utf-8-sig",
)