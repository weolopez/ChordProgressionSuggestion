# CLAUDE.md

## Build/Lint/Test Commands
- Setup environment: `python -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run model conversion: `python convert_model.py`
- Validate ONNX model: `python validate_model.py`
- Run tests: `pytest tests/`
- Lint code: `flake8 *.py`
- Type check: `mypy *.py`

## Code Style Guidelines
- Use PEP 8 formatting standards
- Sort imports: standard library, third-party, local
- Use explicit type annotations for function parameters and returns
- Naming: snake_case for variables/functions, PascalCase for classes
- Document functions with docstrings (Google style)
- Error handling: use try/except blocks with specific exceptions
- Maximum line length: 88 characters
- Use f-strings for string formatting