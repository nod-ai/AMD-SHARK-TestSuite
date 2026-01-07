from pathlib import Path
import json
import sys

json_path = Path(sys.argv[1])
out_path = Path("regression_models.txt")

data = json.loads(json_path.read_text())

with out_path.open("w") as f:
    for model_name in data.keys():
        f.write(f"{model_name}\n")

print(f"Wrote {len(data)} models to {out_path}")

