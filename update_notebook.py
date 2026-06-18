import json

notebook_path = "Acoustic_Wave_Simulation_on_Torus.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Look for the cell that contains the old generate_dataset or run_fast_pipeline
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def run_fast_pipeline" in source or "def generate_multi_rollout_dataset" in source:
            print("Found the pipeline cell!")
            # We will just append the new cells or let's print some info
            pass

