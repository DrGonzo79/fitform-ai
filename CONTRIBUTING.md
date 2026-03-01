# Contributing to FitForm AI

Thank you for your interest in contributing!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dev dependencies: `pip install -r backend/requirements.txt pytest ruff httpx`
4. Copy `.env.example` to `.env` and configure

## Branching Strategy

- `main` — Production-ready code
- `develop` — Integration branch
- `feature/*` — New features
- `fix/*` — Bug fixes

## Code Standards

- **Formatter**: `ruff format`
- **Linter**: `ruff check`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all modules, classes, and public methods

## Testing

```bash
pytest tests/ -v --tb=short
```

Coverage target: 80% for core modules (`exercise_classifier.py`, `rom_calculator.py`, API endpoints).

## Pull Request Process

1. Create a feature branch from `develop`
2. Write tests for new functionality
3. Ensure all tests pass and linting is clean
4. Update documentation if adding new endpoints or features
5. Submit PR with a clear description of changes

## Code Review Checklist

- [ ] Tests pass
- [ ] Linting clean (`ruff check` and `ruff format --check`)
- [ ] Docstrings present
- [ ] No hardcoded credentials or secrets
- [ ] API changes documented in `docs/API.md`
