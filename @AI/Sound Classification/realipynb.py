# Let's read the content of the provided notebook file
import nbformat

with open("/Users/user/HOUSE/@DUNE/@AI/Sound Classification/ai_0014_Analyze.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# Extract and print the code cells from the notebook
code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
code_contents = [cell.source for cell in code_cells]

code_contents
