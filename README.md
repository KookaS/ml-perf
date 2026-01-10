# ML performance

Resources for succeeding as a ML performance engineer.

## Build and run locally

First setup uv and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt jupyter ipykernel
```

Then make sure the python scripts work (if present as a file):
```bash
uv run python -m code.<script_path>
```

Then launch jupyter:
```bash
make jupyter
```

Then serve the book:
```bash
make serve
```