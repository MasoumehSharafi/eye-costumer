from edsai_client import (
    get_all_products,
    get_all_inventory,
    get_catalog,
    find_products,
)


print("\n=== COMPLETE PRODUCT READ ===")

products = get_all_products()

print("Total variants:", len(products))


print("\n=== COMPLETE INVENTORY READ ===")

inventory = get_all_inventory()

print("Total inventory records:", len(inventory))


print("\n=== JOIN PRODUCT + INVENTORY ===")

catalog = get_catalog()

print("Total joined variants:", len(catalog))


print("\n=== FIRST CLEAN PRODUCT ===")

if catalog:
    product = catalog[0]

    print("Title:", product["title"])
    print("Variant:", product["variant_title"])
    print("Price:", product["retail_price"])
    print("Available:", product["available"])
    print("Vendor:", product["vendor"])

    print("\nIngredients:")
    print(
        product["metafields"].get(
            "ingredients"
        )
    )

    print("\nBenefits:")
    print(
        product["metafields"].get(
            "accordion_benefits"
        )
    )

    print("\nExpiry:")
    print(
        product["metafields"].get(
            "expiry_date"
        )
    )


print("\n=== SEARCH TEST ===")

matches = find_products(
    "Thealoz Duo",
    catalog=catalog,
)

for product in matches[:5]:
    print(
        product["title"],
        "|",
        product["variant_title"],
        "| price:",
        product["retail_price"],
        "| available:",
        product["available"],
    )