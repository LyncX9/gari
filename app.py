import streamlit as st
import nbformat
from nbclient import NotebookClient

st.title("🧬 Genetic Algorithm Image Reproducer")

notebook_filename = "gari.ipynb"

st.info("⏳ Running notebook, please wait...")

with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook
client = NotebookClient(nb)
client.execute()

st.success("✅ Notebook executed successfully!")