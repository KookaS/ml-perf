SHELL := /bin/bash

# Run this when you are writing locally
serve:
	mdbook serve

# Run this before you push to GitHub (head.hbs is already prod)
build:
	mdbook build

# Shortcut to run the python server easily
jupyter:
	uv run python -m ipykernel install --user --name=ml-perf --display-name "Python (ML Perf)"
	uv run -- jupyter server --no-browser \
		--ServerApp.ip="0.0.0.0" \
		--ServerApp.allow_origin_pat=".*" \
		--ServerApp.allow_remote_access=True \
		--ServerApp.token="" \
		--ServerApp.disable_check_xsrf=True