# ML performance

Resources for succeeding as a ML performance engineer.

## Build and run locally

First setup uv and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt jupyter ipykernel
```

Then make sure the python scripts work:
```bash
uv run python -m code.distributed.tp
```

Then launch jupyter:
```bash
make run jupyter
```

Then serve the book:
```bash
make serve
```