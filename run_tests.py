"""Minimal test runner — works without pytest.

Discovers test_* functions, runs them, reports pass/fail.
For real development, just use `pytest`.
"""
import sys
import traceback
import tempfile
import inspect
from pathlib import Path


# Replicate pytest's tmp_path fixture
class _TmpPath:
    def __init__(self):
        self._dir = tempfile.mkdtemp()

    def __truediv__(self, name):
        return Path(self._dir) / name

    def cleanup(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)


# Replicate pytest.raises
class _RaisesContext:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but no exception raised")
        if not issubclass(exc_type, self.exc_type):
            return False  # Wrong type, let it propagate
        if self.match:
            import re
            if not re.search(self.match, str(exc_value)):
                raise AssertionError(
                    f"Exception message {str(exc_value)!r} does not match {self.match!r}"
                )
        return True  # Swallow


class _PytestShim:
    @staticmethod
    def raises(exc_type, match=None):
        return _RaisesContext(exc_type, match=match)

    @staticmethod
    def fixture(func):
        # Just mark it; we handle resolution manually
        func._is_fixture = True
        return func


sys.modules["pytest"] = _PytestShim()
import pytest  # noqa: E402

# Now load the test module
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import tests.test_pipeline as test_module  # noqa: E402

# Resolve fixtures by name and run tests
fixtures = {
    name: func
    for name, func in vars(test_module).items()
    if callable(func) and getattr(func, "_is_fixture", False)
}

def call_with_fixtures(func):
    """Inspect func signature, supply fixtures by name."""
    sig = inspect.signature(func)
    kwargs = {}
    tmp_paths_to_cleanup = []
    for param_name in sig.parameters:
        if param_name == "tmp_path":
            tp = _TmpPath()
            kwargs[param_name] = Path(tp._dir)
            tmp_paths_to_cleanup.append(tp)
        elif param_name in fixtures:
            # Run the fixture
            fixture_func = fixtures[param_name]
            fixture_kwargs = {}
            for sub_name in inspect.signature(fixture_func).parameters:
                if sub_name == "tmp_path":
                    tp = _TmpPath()
                    fixture_kwargs[sub_name] = Path(tp._dir)
                    tmp_paths_to_cleanup.append(tp)
            kwargs[param_name] = fixture_func(**fixture_kwargs)
    try:
        return func(**kwargs)
    finally:
        for tp in tmp_paths_to_cleanup:
            tp.cleanup()

tests = [
    (name, func)
    for name, func in vars(test_module).items()
    if name.startswith("test_") and callable(func)
]

passed = 0
failed = 0
failures = []

for name, func in tests:
    try:
        # Redirect output during tests to keep output clean
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            call_with_fixtures(func)
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        failures.append((name, e, traceback.format_exc()))
        failed += 1

print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")

if failures:
    print("\n--- FAILURES ---")
    for name, err, tb in failures:
        print(f"\n{name}:")
        print(tb)
    sys.exit(1)
