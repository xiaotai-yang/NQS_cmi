import nbformat

# Load the notebook
with open("plot.ipynb", "r") as f:
    notebook = nbformat.read(f, as_version=4)

# Save the notebook to ensure it's properly formatted
with open("plot_.ipynb", "w") as f:
    nbformat.write(notebook, f)

print("Notebook fixed and saved as your_notebook_fixed.ipynb")
