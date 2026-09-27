import streamlit as st

from answer_generator import generate_response
from retrieval_backend import retrieve_passages


st.set_page_config(
    page_title="EyeDropShop Brain",
    page_icon="👁️",
)

st.title("EyeDropShop Brain")
st.caption("Local bilingual product-information prototype")

language_label = st.selectbox(
    "Language / Langue",
    ["English", "Français"],
)

language = "en" if language_label == "English" else "fr"

if language == "fr":
    question = st.text_input("Posez votre question sur un produit :")
    button_text = "Demander"
else:
    question = st.text_input("Ask a question about a product:")
    button_text = "Ask"

if st.button(button_text):
    if not question.strip():
        if language == "fr":
            st.warning("Veuillez saisir une question.")
        else:
            st.warning("Please enter a question.")
    else:
        with st.spinner(
            "Génération de la réponse..."
            if language == "fr"
            else "Generating answer..."
        ):
            passages, matched_product = retrieve_passages(
                question,
                language,
            )

            response = generate_response(
                question=question,
                language=language,
                passages=passages,
            )

            answer = response["answer"]
            status = response["status"]

            st.caption(f"Status: `{status}`")
            # st.write(answer)

        if matched_product:
            label = (
                "Produit détecté"
                if language == "fr"
                else "Detected product"
            )
            st.caption(f"{label}: {matched_product.product_title}")

        st.subheader("Réponse" if language == "fr" else "Answer")
        st.write(answer)

        if passages:
            st.subheader(
                "Sources récupérées"
                if language == "fr"
                else "Retrieved sources"
            )

            for index, passage in enumerate(passages, start=1):
                metadata = passage["metadata"]

                source_name = (
                    metadata.get("product_title")
                    or metadata.get("title")
                    or metadata.get("product_handle")
                    or metadata.get("source")
                )

                with st.expander(
                    f"[Source {index}] {source_name} — {metadata['section']}"
                ):
                    st.write(passage["text"])
                    st.write(f"Distance: {passage['distance']:.4f}")

                    if metadata.get("url"):
                        st.markdown(f"[Product page]({metadata['url']})")