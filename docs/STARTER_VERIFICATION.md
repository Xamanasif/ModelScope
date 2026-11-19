# Actual starter verification

Executed: `python -m unittest discover -s tests -v`
Result: 8/8 tests passed on Python 3.12/Linux.

Checks: predictor exclusion; duplicate IDs; invalid target; disjoint, complete,
reproducible splits; known metric results; bad probabilities; single-class error;
API health/status, starter HTML and absent prediction route.

A first run exposed fractional split rounding (41 rather than 40 calibration
records for 400 customers). Fixed the implementation using preallocated integer
counts. Preserved the substantive test expectation.

The environment emitted a Starlette TestClient/httpx deprecation warning. API
tests passed; Claude should verify current supported test-client dependencies
and pin them during completion. Tested direct versions are in requirements-tested.txt.

Not run: official-data acquisition, actual training/calibration, final holdout
evaluation, PowerShell execution, UI browser interaction, production deployment.
The test fixtures are constructed; no real model findings are available yet.
