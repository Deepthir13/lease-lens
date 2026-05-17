"""
utils.py — Shared utilities for Lease Lens.

Includes:
  - Environment helpers (get_ollama_model, get_state)
  - Illegal clause scanner (scan_illegal_clauses, summarize_scan)
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "mistral")


def get_state() -> str:
    return os.getenv("STATE", "CA")


# ---------------------------------------------------------------------------
# Illegal clause definitions
# ---------------------------------------------------------------------------

COMMON_ILLEGAL_CLAUSES: list[dict] = [
    {
        "id": "security_deposit_waiver",
        "description": "Waiving security deposit return timeline",
        "pattern": r"(waive[sd]?|forfeit[s]?|no[t]?\s+(required|obligated)\s+to\s+return|not\s+responsible\s+for\s+returning).{0,80}(security deposit|deposit)",
        "why_illegal": (
            "Most states require landlords to return security deposits within a set deadline "
            "(e.g., 14–21 days) with an itemized statement. A clause waiving or extending this "
            "obligation beyond statutory limits is unenforceable."
        ),
        "severity": "high",
        "states": "all",
    },
    {
        "id": "esa_prohibition",
        "description": "Prohibiting emotional support animals (Fair Housing Act violation)",
        "pattern": r"(no|prohibit|not\s+(allowed|permitted|accepted)).{0,60}(animal|pet|dog|cat).{0,80}(emotional\s+support|assistance|service|esa)",
        "why_illegal": (
            "The Fair Housing Act requires landlords to make reasonable accommodations for "
            "tenants with disabilities, including allowing emotional support animals regardless "
            "of a 'no pets' policy. Blanket bans on ESAs violate federal law."
        ),
        "severity": "high",
        "states": "all",
    },
    {
        "id": "entry_notice_waiver",
        "description": "Waiving landlord entry notice requirement",
        "pattern": r"(landlord|lessor|owner|management).{0,60}(enter|access|inspect).{0,60}(without|no|any).{0,30}(notice|notification|warning|prior)",
        "why_illegal": (
            "Most states (e.g., CA §1954, NY RPL §235-b) require landlords to give advance "
            "notice (typically 24 hours) before entering. A lease clause purporting to waive "
            "this right is void as against public policy."
        ),
        "severity": "high",
        "states": "all",
    },
    {
        "id": "auto_renewal_no_notice",
        "description": "Automatic lease renewal without written notice to tenant",
        "pattern": r"(automatically|auto).{0,30}(renew|extend|convert).{0,60}(without|unless).{0,60}(written|notice|notif)",
        "why_illegal": (
            "Many states prohibit automatic renewal clauses unless the tenant received "
            "clear, conspicuous written notice of the clause before signing. Buried "
            "auto-renewal clauses are unenforceable in several jurisdictions."
        ),
        "severity": "medium",
        "states": "all",
    },
    {
        "id": "no_repair_responsibility",
        "description": "Landlord not responsible for any repairs",
        "pattern": r"(landlord|lessor|owner|management).{0,60}(not (responsible|liable|required|obligated)|no (obligation|duty|responsibility)).{0,60}(repair|maintain|fix|upkeep|habitab)",
        "why_illegal": (
            "The implied warranty of habitability (recognized in all 50 states) requires "
            "landlords to maintain rental units in livable condition. A clause disclaiming "
            "all repair responsibility is void and unenforceable."
        ),
        "severity": "high",
        "states": "all",
    },
    {
        "id": "one_sided_attorney_fees",
        "description": "Tenant pays all attorney fees regardless of outcome",
        "pattern": r"tenant.{0,80}(pay[s]?|responsible for|liable for|bear[s]?).{0,60}(all|any|legal|attorney|counsel).{0,30}(fee[s]?|cost[s]?|expense[s]?)",
        "why_illegal": (
            "One-sided attorney fee clauses that require only the tenant to pay fees "
            "regardless of who wins are unenforceable in many states (e.g., CA Civil Code "
            "§1717 implies reciprocity; NY RPL §234 implies a reciprocal right for tenants). "
            "Courts routinely strike these as unconscionable."
        ),
        "severity": "medium",
        "states": "all",
    },
    {
        "id": "jury_trial_waiver",
        "description": "Waiving right to jury trial",
        "pattern": r"(waive[sd]?|give[s]? up|relinquish|forfeit).{0,60}(right to|any right).{0,40}(jury|jury trial|trial by jury)",
        "why_illegal": (
            "Pre-dispute jury trial waivers in residential leases are prohibited or "
            "unenforceable in many states (including NY, NJ, and CA) because they are "
            "considered contracts of adhesion that deprive tenants of a fundamental right."
        ),
        "severity": "high",
        "states": ["NY", "NJ", "CA", "WA", "IL", "MA", "CT"],
    },
    {
        "id": "excessive_late_fee",
        "description": "Excessive late fee (over 10% of monthly rent in most states)",
        "pattern": r"late (fee|charge|penalty).{0,60}(\$\s?\d{3,}|\d{1,3}\s*%)",
        "why_illegal": (
            "Most states cap late fees at 5–10% of the monthly rent or a flat maximum "
            "(e.g., CA caps at $100 for the first violation, TX at 12%, WA at a reasonable "
            "amount). Late fees above the statutory cap are unenforceable as liquidated "
            "damages penalties."
        ),
        "severity": "medium",
        "states": "all",
    },
]


# ---------------------------------------------------------------------------
# Clause scanner
# ---------------------------------------------------------------------------

def _split_into_clauses(lease_text: str) -> list[str]:
    """
    Split lease text into individual clauses for analysis.

    Strategy (in order):
      1. Split on numbered section headers (e.g., "1.", "2.", "Section 3", "ARTICLE IV")
      2. Fall back to double-newline paragraph breaks
      3. If still single block, split on sentences
    """
    # Try numbered sections / article headers
    section_pattern = re.compile(
        r"(?:^|\n)(?:(?:section|article|clause|paragraph)\s+\w+\.?|(?:\d+|[ivxlcdm]+)\.)\s+",
        re.IGNORECASE | re.MULTILINE,
    )
    parts = section_pattern.split(lease_text)
    if len(parts) > 2:
        return [p.strip() for p in parts if p.strip()]

    # Paragraph breaks
    parts = re.split(r"\n{2,}", lease_text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    # Sentence splitting as last resort
    parts = re.split(r"(?<=[.!?])\s+", lease_text)
    return [p.strip() for p in parts if p.strip()]


def _applies_to_state(clause_def: dict, state: str) -> bool:
    """Return True if the clause pattern is relevant for the given state."""
    states = clause_def.get("states", "all")
    if states == "all":
        return True
    return state.upper() in [s.upper() for s in states]


def scan_illegal_clauses(lease_text: str, state: str) -> list[dict]:
    """
    Scan lease text for potentially illegal or unenforceable clauses.

    Args:
        lease_text: Full plain-text content of the lease.
        state:      Two-letter state abbreviation (used for state-specific rules).

    Returns:
        List of findings, each a dict:
        {
            "clause_text":  str,   # the matching clause/paragraph text
            "issue":        str,   # human-readable clause type name
            "why_illegal":  str,   # explanation of the legal problem
            "severity":     str,   # "high" or "medium"
        }
    """
    clauses = _split_into_clauses(lease_text)
    findings = []
    seen_ids = set()  # avoid duplicate findings for the same pattern

    for clause in clauses:
        clause_lower = clause.lower()
        for defn in COMMON_ILLEGAL_CLAUSES:
            if defn["id"] in seen_ids:
                continue
            if not _applies_to_state(defn, state):
                continue
            if re.search(defn["pattern"], clause_lower, re.IGNORECASE | re.DOTALL):
                findings.append({
                    "clause_text": clause.strip(),
                    "issue": defn["description"],
                    "why_illegal": defn["why_illegal"],
                    "severity": defn["severity"],
                })
                seen_ids.add(defn["id"])

    return findings


def summarize_scan(scan_results: list[dict]) -> str:
    """
    Generate a plain-English summary of illegal clause scan results.

    Args:
        scan_results: The list returned by scan_illegal_clauses().

    Returns:
        A formatted summary string.
    """
    if not scan_results:
        return (
            "Good news — no obviously problematic clauses were detected in your lease. "
            "This scan checks for common patterns only. Consider having a tenant rights "
            "attorney review the full document to be certain."
        )

    high = [r for r in scan_results if r["severity"] == "high"]
    medium = [r for r in scan_results if r["severity"] == "medium"]

    count = len(scan_results)
    noun = "clause" if count == 1 else "clauses"

    lines = [
        f"We found {count} potentially unenforceable {noun} in your lease "
        f"({len(high)} high severity, {len(medium)} medium severity).\n"
    ]

    for i, finding in enumerate(scan_results, start=1):
        badge = "🔴 HIGH" if finding["severity"] == "high" else "🟡 MEDIUM"
        lines.append(f"{i}. [{badge}] {finding['issue']}")
        lines.append(f"   Why it may be illegal: {finding['why_illegal']}")
        excerpt = finding["clause_text"][:200].replace("\n", " ")
        if len(finding["clause_text"]) > 200:
            excerpt += "..."
        lines.append(f"   Lease text: \"{excerpt}\"")
        lines.append("")

    lines.append(
        "Note: This is an automated pattern scan, not legal advice. "
        "Consult a tenant rights attorney to confirm enforceability in your jurisdiction."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_LEASE = """
    RESIDENTIAL LEASE AGREEMENT

    1. RENT AND LATE FEES
    Monthly rent is $2,000 due on the first of each month.
    A late fee of $300 will be charged if rent is not received by the 5th.

    2. SECURITY DEPOSIT
    Tenant shall pay a security deposit of $4,000. Landlord is not required
    to return the security deposit and may retain it for any reason at
    landlord's sole discretion.

    3. ENTRY
    Landlord or landlord's agents may enter the premises at any time without
    prior notice to tenant for any reason including inspection, repairs, or
    showing the unit to prospective tenants.

    4. REPAIRS
    Landlord is not responsible for any repairs, maintenance, or upkeep of
    the premises. Tenant accepts the unit in as-is condition.

    5. ATTORNEY FEES
    In any legal proceeding arising from this lease, tenant shall pay all
    attorney fees and legal costs incurred by landlord.

    6. JURY TRIAL WAIVER
    Tenant hereby waives any right to a jury trial in any dispute
    arising out of or related to this lease agreement.

    7. PETS AND ANIMALS
    No animals are allowed on the premises. Emotional support animals and
    service animals are not permitted without express written consent.

    8. AUTOMATIC RENEWAL
    This lease will automatically renew for successive one-year terms
    without written notice to tenant unless tenant provides 60 days notice.
    """

    print("Running illegal clause scan on sample lease...\n")
    results = scan_illegal_clauses(SAMPLE_LEASE, state="CA")
    print(summarize_scan(results))
    print(f"\nTotal findings: {len(results)}")
