# SmartGrocy Refactoring Plan

## 📋 Overview

This document outlines the refactoring plan for the SmartGrocy project to improve code organization, maintainability, and extensibility.

## 🎯 Goals

1. **Reduce Code Duplication**: Create base classes and shared utilities
2. **Unify Entry Points**: Single CLI interface instead of multiple scripts
3. **Improve Configuration**: Type-safe configuration management
4. **Better Error Handling**: Custom exceptions and error recovery
5. **Type Safety**: Add comprehensive type hints
6. **Modularity**: Better separation of concerns

## 📁 New Structure

```
src/
├── core/                    # Core abstractions and base classes
│   ├── __init__.py
│   ├── base.py             # BasePipeline, BaseModule, BaseConfig
│   ├── exceptions.py       # Custom exceptions
│   └── config_manager.py   # Configuration management
├── cli/                     # Unified CLI
│   ├── __init__.py
│   └── main.py             # Main CLI entry point
├── features/                # Feature engineering (unchanged)
├── modules/                 # Business modules (unchanged)
├── pipelines/               # Pipeline stages (refactored)
└── utils/                   # Utilities (unchanged)
```

## 🔄 Refactoring Steps

### Phase 1: Core Infrastructure ✅
- [x] Create `src/core/base.py` with base classes
- [x] Create `src/core/exceptions.py` with custom exceptions
- [x] Create `src/core/config_manager.py` for configuration
- [x] Create `src/cli/main.py` for unified CLI

### Phase 2: Pipeline Refactoring
- [ ] Refactor pipelines to inherit from `BasePipeline`
- [ ] Consolidate orchestrator files (merge v1 and v2)
- [ ] Improve error handling in pipelines
- [ ] Add type hints to pipeline functions

### Phase 3: Module Refactoring
- [ ] Refactor business modules to inherit from `BaseModule`
- [ ] Add input/output validation
- [ ] Improve error handling

### Phase 4: Configuration Migration
- [ ] Migrate from `src/config.py` to `ConfigManager`
- [ ] Update all imports
- [ ] Add configuration validation

### Phase 5: Entry Point Consolidation
- [ ] Update `main.py` to use new CLI
- [ ] Deprecate old entry points (with warnings)
- [ ] Update documentation

## 📝 Migration Guide

### For Developers

#### Using New CLI
```bash
# Old way
python main.py pipeline --full-data
python run_business_modules.py --forecasts file.csv

# New way
python -m src.cli.main pipeline --full-data
python -m src.cli.main business --forecasts file.csv
```

#### Using New Configuration
```python
# Old way
from src.config import ACTIVE_DATASET, DATASET_CONFIGS

# New way
from src.core.config_manager import ConfigManager
manager = ConfigManager()
config = manager.get_active_dataset_config()
```

#### Using Base Classes
```python
# Old way
def my_pipeline():
    # Setup code
    # Run code
    # Cleanup code

# New way
from src.core.base import BasePipeline, BaseConfig

class MyPipeline(BasePipeline):
    def run(self, **kwargs):
        # Run code
        pass
```

## ⚠️ Breaking Changes

1. **CLI Changes**: Old entry points will be deprecated
2. **Configuration**: `src/config.py` will be gradually replaced
3. **Imports**: Some imports will change

## 🧪 Testing Strategy

1. Unit tests for base classes
2. Integration tests for pipelines
3. CLI tests for all commands
4. Configuration validation tests

## 📅 Timeline

- **Week 1**: Core infrastructure (Phase 1) ✅
- **Week 2**: Pipeline refactoring (Phase 2)
- **Week 3**: Module refactoring (Phase 3)
- **Week 4**: Configuration migration (Phase 4)
- **Week 5**: Entry point consolidation (Phase 5)

## 🔍 Code Quality Improvements

- Type hints coverage: Target 80%+
- Test coverage: Target 70%+
- Documentation: All public APIs documented
- Linting: Pass all linters (ruff, mypy, pylint)

