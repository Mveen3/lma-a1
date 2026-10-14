"""
Execute RAG_Baseline_vs_Reranking.ipynb end-to-end and save outputs.
"""

import io
import sys
import nbformat
from nbformat.v4 import new_output

notebook_file = "RAG_Baseline_vs_Reranking.ipynb"
with open(notebook_file, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

global_env = {}
exec_count = 1

print(f"Starting execution of {notebook_file} ({len(nb.cells)} cells)...")

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"\n--- Executing Code Cell {exec_count} (Cell Index {idx}) ---")
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec(cell.source, global_env)
            sys.stdout = old_stdout
            out_str = redirected_output.getvalue()
            cell.outputs = [new_output(output_type="stream", name="stdout", text=out_str)]
            cell.execution_count = exec_count
            print(f"Cell {exec_count} executed successfully.")
            # Print preview
            preview = out_str[:300].replace("\n", " | ")
            print(f"Output Preview: {preview}...")
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error executing cell {exec_count}: {e}")
            raise e
        exec_count += 1

with open(notebook_file, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"\nNotebook execution complete! All outputs successfully saved into: {notebook_file}")
