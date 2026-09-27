from pathlib import Path

import pandas as pd


INPUT_FOLDER = Path("knowledge/converted")
OUTPUT_FOLDER = Path("inspection_results")

OUTPUT_FOLDER.mkdir(exist_ok=True)

records = []

for file_path in sorted(INPUT_FOLDER.rglob("*.txt")):
    text = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    relative_path = file_path.relative_to(
        INPUT_FOLDER
    )

    records.append(
        {
            "file": str(relative_path),
            "characters": len(text),
            "words": len(text.split()),
            "nonempty_lines": len(lines),
            "preview": "\n".join(lines[:80]),
        }
    )

inspection = pd.DataFrame(records)

inspection.to_csv(
    OUTPUT_FOLDER / "support_document_inspection.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "Saved: "
    "inspection_results/support_document_inspection.csv"
)