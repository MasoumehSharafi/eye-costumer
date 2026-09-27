import pandas as pd

files = [
    "knowledge/products/Active Products CA EN.xlsx",
    "knowledge/products/Active Products CA FR.xlsx",
]

for file in files:
    excel_file = pd.ExcelFile(file)

    print(f"\nFile: {file}")
    print("Sheets:", excel_file.sheet_names)

    for sheet in excel_file.sheet_names:
        data = pd.read_excel(file, sheet_name=sheet)

        print(f"\nSheet: {sheet}")
        print("Number of rows:", len(data))
        print("Columns:")
        for column in data.columns:
            print("-", column)