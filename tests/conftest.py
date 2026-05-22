"""
conftest.py — CI-friendly fixtures for Lease Lens tests.

Mocks out external services so the full test suite runs without:
  - A live Ollama server (ollama.chat → canned answer)
  - Network access  (requests.get → stub statute text)
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub responses
# ---------------------------------------------------------------------------

_MOCK_ANSWER = (
    "Your lease says the late fee policy requires payment by the 5th of each month. "
    "Your state law says landlords must maintain rental units in habitable condition "
    "and provide at least 24-hour written notice before entering. "
    "Note: This is informational only. Consult a tenant rights attorney for your specific situation."
)

_MOCK_STATUTE_HTML = """\
<html><body>
<h1>California Tenant Rights and Habitability</h1>

<h2>Section 1 — Security Deposits</h2>
<p>Landlords must return security deposits within 21 days of move-out.
An itemized statement of deductions must accompany any withholding.
Penalties for wrongful withholding may be up to twice the deposit amount
under California Civil Code 1950.5.</p>

<h2>Section 2 — Landlord Entry</h2>
<p>Landlords must provide at least 24-hour written notice before entering.
Exceptions apply to genuine emergencies and unit abandonment.</p>

<h2>Section 3 — Repairs and Habitability</h2>
<p>Landlords must maintain rental units in habitable condition including
functioning heat, hot water, and weatherproofing per Civil Code 1941.
Tenants may use the repair-and-deduct remedy for repairs up to one month rent.</p>

<h2>Section 4 — Notice Requirements</h2>
<p>Month-to-month tenants require 30 days written notice to terminate.
Annual leases require appropriate notice at least 60 days before end of term.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# Autouse fixtures — applied to every test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_ollama():
    """Replace ollama.chat with a deterministic stub so no LLM is needed."""
    stub = {"message": {"content": _MOCK_ANSWER}}
    with patch("ollama.chat", return_value=stub):
        yield


@pytest.fixture(autouse=True)
def _mock_http():
    """Replace requests.get with a stub so state-law scraping works offline."""
    mock_resp = MagicMock()
    mock_resp.text = _MOCK_STATUTE_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200
    with patch("requests.get", return_value=mock_resp):
        yield
