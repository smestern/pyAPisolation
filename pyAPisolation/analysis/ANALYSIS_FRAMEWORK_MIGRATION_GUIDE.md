# Migration Guide: Analysis Framework Reorganization

## What Changed

The analysis framework has been moved from `pyAPisolation.gui.analysis_modules` to its own dedicated package at `pyAPisolation.analysis`. This provides better separation of concerns and makes the framework more reusable.

## Module Structure Changes

### Before (Old Structure)
```
pyAPisolation/gui/analysis_modules.py    # Everything in one file
```

### After (New Structure)
```
pyAPisolation/analysis/
├── __init__.py              # Main package interface
├── base.py                  # Abstract AnalysisModule class
├── registry.py              # AnalysisRegistry class  
├── builtin_modules.py       # Legacy spike & subthreshold modules
└── utilities.py             # Registration utilities & decorators
```

## Import Changes

### Old Imports (No longer work)
```python
# These imports will fail
from pyAPisolation.gui.analysis_modules import AnalysisModule
from pyAPisolation.gui.analysis_modules import analysis_registry
from pyAPisolation.gui.analysis_modules import register_analysis_module
```

### New Imports (Use these instead)
```python
# Core classes
from pyAPisolation.analysis import AnalysisModule
from pyAPisolation.analysis import analysis_registry

# Utility functions
from pyAPisolation.analysis import register_analysis_module
from pyAPisolation.analysis import register_analysis_with_tab
from pyAPisolation.analysis import list_available_analyses
from pyAPisolation.analysis import get_analysis_module
from pyAPisolation.analysis import analysis_module  # decorator

# Built-in modules (if needed directly)
from pyAPisolation.analysis import SpikeAnalysisModule
from pyAPisolation.analysis import SubthresholdAnalysisModule
```

## Migration Steps

### For Existing Custom Analysis Modules

1. **Update Imports**: Change all imports from `pyAPisolation.gui.analysis_modules` to `pyAPisolation.analysis`

2. **No Code Changes**: Your existing analysis module implementations don't need to change, just the imports

3. **Test**: Run your code to ensure everything still works

### Example Migration

**Before:**
```python
from pyAPisolation.gui.analysis_modules import AnalysisModule, register_analysis_module

class MyAnalysis(AnalysisModule):
    # ... implementation stays the same ...

# Register it
register_analysis_module(MyAnalysis)
```

**After:**
```python
from pyAPisolation.analysis import AnalysisModule, register_analysis_module

class MyAnalysis(AnalysisModule):
    # ... implementation stays the same ...

# Register it
register_analysis_module(MyAnalysis)
```

## Benefits of New Structure

1. **Better Organization**: Analysis framework is separate from GUI code
2. **Cleaner Imports**: More logical package structure
3. **Reusability**: Analysis modules can be used without GUI dependencies
4. **Maintainability**: Easier to find and modify specific components
5. **Future-Proof**: Better foundation for framework expansion

## Backward Compatibility

- **GUI Still Works**: The GUI automatically uses the new framework location
- **Legacy Analysis**: All existing spike and subthreshold analysis functionality is preserved
- **No Breaking Changes**: Existing workflows and outputs remain identical

## Files That Were Updated

- **Created**: `pyAPisolation/analysis/` package with new structure
- **Updated**: `pyAPisolation/gui/spikeFinder.py` to use new imports
- **Updated**: All example scripts to use new imports
- **Removed**: `pyAPisolation/gui/analysis_modules.py` (replaced by package)

## Quick Migration Script

If you have multiple files to migrate, you can use this PowerShell script:

```powershell
# Find and replace imports in Python files
Get-ChildItem -Path "." -Include "*.py" -Recurse | ForEach-Object {
    (Get-Content $_.FullName) -replace 
    'from pyAPisolation.gui.analysis_modules import', 
    'from pyAPisolation.analysis import' | 
    Set-Content $_.FullName
}
```

Or this Python script:

```python
import os
import re

def migrate_imports(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Replace the import
                new_content = re.sub(
                    r'from pyAPisolation\.gui\.analysis_modules import',
                    'from pyAPisolation.analysis import',
                    content
                )
                
                if new_content != content:
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print(f"Updated: {filepath}")

# Run migration
migrate_imports(".")
```

## Verification

After migration, verify everything works:

1. **Import Test**:
   ```python
   from pyAPisolation.analysis import AnalysisModule, analysis_registry
   print("✓ New imports work")
   ```

2. **Registry Test**:
   ```python
   from pyAPisolation.analysis import list_available_analyses
   modules = list_available_analyses()
   print(f"✓ Available modules: {list(modules.keys())}")
   ```

3. **GUI Test**: Run the GUI and verify all analysis functions work as expected

## Need Help?

- Check the updated examples in `add_custom_analysis_example.py`
- Run the test script: `python test_registration_utilities.py`
- Review the updated documentation in `MODULAR_ANALYSIS_README.md`
