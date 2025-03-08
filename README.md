# Agent Cookbook

## Setup

1. Install uv:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv sync
```

## Development

Project configuration is managed in `pyproject.toml`. This includes:
- Project metadata (name, version, authors)
- Python version requirements
- All project dependencies with their versions

- To add new dependencies:
```bash
# Add the dependency to pyproject.toml under [project.dependencies], then run:
uv sync
```

- To update dependencies:
```bash
uv sync --upgrade
```

Note: This project uses the Aliyun PyPI mirror for faster package downloads in China. If you're outside China, you can use the official PyPI repository by default.
