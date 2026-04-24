#!/bin/bash
mkdir -p canonical_ar/{data/{raw,synthetic,processed},src/{data,models,utils,eval},configs,scripts,notebooks,outputs/{checkpoints,logs,renders}}
touch canonical_ar/src/__init__.py
touch canonical_ar/src/data/__init__.py
touch canonical_ar/src/models/__init__.py
touch canonical_ar/src/utils/__init__.py
touch canonical_ar/src/eval/__init__.py
echo "Project structure created"
