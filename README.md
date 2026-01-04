# ML performance

Resources for succeeding as a ML performance engineer.

## Build and run on WSL

First setup uv:
```bash
uv venv
uv pip install -r requirements.txt jupyter ipykernel
```

Then make sure the python scripts work:
```bash
uv run python -m code.distributed.tp
```

Then install the python dependencies and launch jupyter:
```bash
make run jupyter
```

Then build and serve the book locally:
```bash
make serve
```