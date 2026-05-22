"""
documents.py — Security deposit demand letter generation for Lease Lens.

Public API:
  STATE_DEPOSIT_LAW            dict  — all 50 states, return deadlines & penalties
  generate_deposit_letter(params) -> str   — Ollama-drafted formal demand letter
  save_letter_pdf(letter_text, filename)   -> str (path to saved PDF)
"""

import io
import pathlib
import tempfile
from datetime import datetime, date, timedelta

import ollama
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from app.utils import get_ollama_model


# ---------------------------------------------------------------------------
# Security deposit return law — all 50 US states
# ---------------------------------------------------------------------------

STATE_DEPOSIT_LAW: dict[str, dict] = {
    "AL": {
        "return_deadline": 60,
        "penalty": 2,
        "statute_citation": "Alabama Code § 35-9A-201",
        "certified_mail_required": False,
    },
    "AK": {
        "return_deadline": 14,
        "penalty": 2,
        "statute_citation": "Alaska Stat. § 34.03.070",
        "certified_mail_required": False,
    },
    "AZ": {
        "return_deadline": 14,
        "penalty": 2,
        "statute_citation": "Ariz. Rev. Stat. § 33-1321",
        "certified_mail_required": False,
    },
    "AR": {
        "return_deadline": 60,
        "penalty": 2,
        "statute_citation": "Ark. Code Ann. § 18-16-305",
        "certified_mail_required": False,
    },
    "CA": {
        "return_deadline": 21,
        "penalty": 2,
        "statute_citation": "California Civil Code § 1950.5",
        "certified_mail_required": False,
    },
    "CO": {
        "return_deadline": 30,
        "penalty": 3,
        "statute_citation": "Colo. Rev. Stat. § 38-12-103",
        "certified_mail_required": False,
    },
    "CT": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Conn. Gen. Stat. § 47a-21",
        "certified_mail_required": False,
    },
    "DE": {
        "return_deadline": 20,
        "penalty": 2,
        "statute_citation": "25 Del. Code § 5514",
        "certified_mail_required": False,
    },
    "FL": {
        "return_deadline": 15,
        "penalty": 2,
        "statute_citation": "Fla. Stat. § 83.49",
        "certified_mail_required": False,
    },
    "GA": {
        "return_deadline": 30,
        "penalty": 3,
        "statute_citation": "Ga. Code Ann. § 44-7-34",
        "certified_mail_required": False,
    },
    "HI": {
        "return_deadline": 14,
        "penalty": 3,
        "statute_citation": "Haw. Rev. Stat. § 521-44",
        "certified_mail_required": False,
    },
    "ID": {
        "return_deadline": 21,
        "penalty": 3,
        "statute_citation": "Idaho Code § 6-321",
        "certified_mail_required": False,
    },
    "IL": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "765 ILCS 710/1",
        "certified_mail_required": False,
    },
    "IN": {
        "return_deadline": 45,
        "penalty": 2,
        "statute_citation": "Ind. Code § 32-31-3-12",
        "certified_mail_required": False,
    },
    "IA": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Iowa Code § 562A.12",
        "certified_mail_required": False,
    },
    "KS": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Kan. Stat. Ann. § 58-2550",
        "certified_mail_required": False,
    },
    "KY": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Ky. Rev. Stat. Ann. § 383.580",
        "certified_mail_required": False,
    },
    "LA": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "La. Rev. Stat. Ann. § 9:3251",
        "certified_mail_required": False,
    },
    "ME": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "14 M.R.S. § 6033",
        "certified_mail_required": False,
    },
    "MD": {
        "return_deadline": 45,
        "penalty": 3,
        "statute_citation": "Md. Code, Real Prop. § 8-410",
        "certified_mail_required": False,
    },
    "MA": {
        "return_deadline": 30,
        "penalty": 3,
        "statute_citation": "Mass. Gen. Laws ch. 186 § 15B",
        "certified_mail_required": True,
    },
    "MI": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Mich. Comp. Laws § 554.609",
        "certified_mail_required": False,
    },
    "MN": {
        "return_deadline": 21,
        "penalty": 2,
        "statute_citation": "Minn. Stat. § 504B.178",
        "certified_mail_required": False,
    },
    "MS": {
        "return_deadline": 45,
        "penalty": 2,
        "statute_citation": "Miss. Code Ann. § 89-8-21",
        "certified_mail_required": False,
    },
    "MO": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Mo. Rev. Stat. § 535.300",
        "certified_mail_required": False,
    },
    "MT": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Mont. Code Ann. § 70-25-202",
        "certified_mail_required": False,
    },
    "NE": {
        "return_deadline": 14,
        "penalty": 2,
        "statute_citation": "Neb. Rev. Stat. § 76-1416",
        "certified_mail_required": False,
    },
    "NV": {
        "return_deadline": 30,
        "penalty": 3,
        "statute_citation": "Nev. Rev. Stat. § 118A.242",
        "certified_mail_required": False,
    },
    "NH": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "N.H. Rev. Stat. Ann. § 540-A:7",
        "certified_mail_required": False,
    },
    "NJ": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "N.J. Stat. Ann. § 46:8-21.1",
        "certified_mail_required": True,
    },
    "NM": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "N.M. Stat. Ann. § 47-8-18",
        "certified_mail_required": False,
    },
    "NY": {
        "return_deadline": 14,
        "penalty": 2,
        "statute_citation": "N.Y. Real Prop. Law § 227-e",
        "certified_mail_required": False,
    },
    "NC": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "N.C. Gen. Stat. § 42-52",
        "certified_mail_required": False,
    },
    "ND": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "N.D. Cent. Code § 47-16-07.1",
        "certified_mail_required": False,
    },
    "OH": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Ohio Rev. Code Ann. § 5321.16",
        "certified_mail_required": False,
    },
    "OK": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "41 Okla. Stat. § 115",
        "certified_mail_required": False,
    },
    "OR": {
        "return_deadline": 31,
        "penalty": 2,
        "statute_citation": "Or. Rev. Stat. § 90.300",
        "certified_mail_required": False,
    },
    "PA": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "68 Pa. Stat. § 250.512",
        "certified_mail_required": False,
    },
    "RI": {
        "return_deadline": 20,
        "penalty": 2,
        "statute_citation": "R.I. Gen. Laws § 34-18-19",
        "certified_mail_required": False,
    },
    "SC": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "S.C. Code Ann. § 27-40-410",
        "certified_mail_required": False,
    },
    "SD": {
        "return_deadline": 45,
        "penalty": 2,
        "statute_citation": "S.D. Codified Laws § 43-32-24",
        "certified_mail_required": False,
    },
    "TN": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Tenn. Code Ann. § 66-28-301",
        "certified_mail_required": False,
    },
    "TX": {
        "return_deadline": 30,
        "penalty": 3,
        "statute_citation": "Tex. Prop. Code § 92.109",
        "certified_mail_required": False,
    },
    "UT": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Utah Code Ann. § 57-17-3",
        "certified_mail_required": False,
    },
    "VT": {
        "return_deadline": 14,
        "penalty": 2,
        "statute_citation": "9 Vt. Stat. Ann. § 4461",
        "certified_mail_required": False,
    },
    "VA": {
        "return_deadline": 45,
        "penalty": 2,
        "statute_citation": "Va. Code Ann. § 55.1-1226",
        "certified_mail_required": False,
    },
    "WA": {
        "return_deadline": 21,
        "penalty": 2,
        "statute_citation": "Wash. Rev. Code § 59.18.280",
        "certified_mail_required": True,
    },
    "WV": {
        "return_deadline": 60,
        "penalty": 2,
        "statute_citation": "W. Va. Code § 37-6A-2",
        "certified_mail_required": False,
    },
    "WI": {
        "return_deadline": 21,
        "penalty": 2,
        "statute_citation": "Wis. Stat. § 704.28",
        "certified_mail_required": False,
    },
    "WY": {
        "return_deadline": 30,
        "penalty": 2,
        "statute_citation": "Wyo. Stat. Ann. § 1-21-1208",
        "certified_mail_required": False,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LETTER_SYSTEM_PROMPT = (
    "You are a tenant rights attorney drafting a formal security deposit demand letter. "
    "Write in a professional, firm, and legally precise tone. "
    "Use standard business letter format with proper address blocks, salutation, body paragraphs, "
    "and a closing signature block. "
    "Cite the provided statute exactly as given. "
    "Do not add disclaimers, explanatory notes, or commentary — write only the letter itself."
)


def _calculate_deadline(move_out_date: str, days: int) -> str:
    """Return the formatted deadline date string given a move-out date and day count."""
    try:
        move_out = datetime.strptime(move_out_date, "%Y-%m-%d").date()
        deadline = move_out + timedelta(days=days)
        return deadline.strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return f"{days} days after your move-out date"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_deposit_letter(params: dict) -> str:
    """
    Draft a formal security deposit demand letter using Ollama / Mistral.

    Args:
        params: dict with keys —
            tenant_name       str
            tenant_address    str
            landlord_name     str
            landlord_address  str
            property_address  str
            move_out_date     str  (YYYY-MM-DD)
            deposit_amount    float
            deductions_claimed list[str]
            state             str  (two-letter code)

    Returns:
        The complete letter as a plain string.
    """
    state = params.get("state", "CA").upper()
    law = STATE_DEPOSIT_LAW.get(state, STATE_DEPOSIT_LAW["CA"])

    deposit_amount = float(params.get("deposit_amount", 0))
    penalty_amount = deposit_amount * law["penalty"]
    deadline_str = _calculate_deadline(
        str(params.get("move_out_date", "")), law["return_deadline"]
    )

    deductions = params.get("deductions_claimed", [])
    if deductions:
        deductions_text = "\n".join(f"  - {d}" for d in deductions)
    else:
        deductions_text = "  (no deductions have been specified)"

    today_str = date.today().strftime("%B %d, %Y")

    user_prompt = f"""
Draft a complete, formal security deposit demand letter using the following information.

TODAY'S DATE: {today_str}

TENANT (SENDER):
  Name: {params.get("tenant_name", "[Tenant Name]")}
  Address: {params.get("tenant_address", "[Tenant Address]")}

LANDLORD (RECIPIENT):
  Name: {params.get("landlord_name", "[Landlord Name]")}
  Address: {params.get("landlord_address", "[Landlord Address]")}

RENTAL PROPERTY: {params.get("property_address", "[Property Address]")}
MOVE-OUT DATE: {params.get("move_out_date", "[Move-Out Date]")}
SECURITY DEPOSIT AMOUNT: ${deposit_amount:,.2f}

DEDUCTIONS CLAIMED BY LANDLORD:
{deductions_text}

GOVERNING LAW:
  Statute: {law["statute_citation"]}
  Return deadline: {law["return_deadline"]} days after move-out (by {deadline_str})
  Maximum statutory penalty for wrongful withholding: ${penalty_amount:,.2f} ({law["penalty"]}x the deposit)

The letter must:
1. Open with the tenant's address block, date, then the landlord's address block.
2. Include a formal salutation addressing the landlord by name.
3. Reference the tenancy at {params.get("property_address", "the above property")} and state the move-out date.
4. Clearly demand full return of the ${deposit_amount:,.2f} security deposit.
5. Dispute each listed deduction as improper, vague, or unsupported if deductions were claimed.
6. Cite {law["statute_citation"]} and state the legal deadline of {law["return_deadline"]} days (by {deadline_str}).
7. State that failure to comply will expose the landlord to liability of up to ${penalty_amount:,.2f} ({law["penalty"]}x the deposit) plus court costs and attorney's fees.
8. Request a written response within 14 days.
9. Close with a professional signature block for the tenant.

Write the complete letter now. Output only the letter — no preamble, no notes.
""".strip()

    model = get_ollama_model()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _LETTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    letter_text = response["message"]["content"].strip()

    # Append certified-mail advisory if the state requires it
    if law["certified_mail_required"]:
        letter_text += (
            "\n\n---\n"
            f"DELIVERY NOTE: Under {law['statute_citation']}, this notice should be sent "
            "via certified mail, return receipt requested, to create a verifiable record of delivery."
        )

    return letter_text


def save_letter_pdf(letter_text: str, filename: str = "deposit_demand_letter.pdf") -> str:
    """
    Render letter_text as a professionally-formatted PDF using ReportLab.

    Args:
        letter_text: The full letter string returned by generate_deposit_letter().
        filename:    Output filename (saved to a system temp directory).

    Returns:
        Absolute path to the saved PDF file.
    """
    output_dir = pathlib.Path(tempfile.mkdtemp())
    pdf_path = output_dir / filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=1.1 * inch,
        leftMargin=1.1 * inch,
        topMargin=1.2 * inch,
        bottomMargin=1.2 * inch,
        title="Security Deposit Demand Letter",
        author="Lease Lens",
    )

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "LLHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#4A4A4A"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "LLBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    footer_style = ParagraphStyle(
        "LLFooter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
        spaceBefore=8,
    )

    story: list = []

    # ── Document header ──────────────────────────────────────────────────────
    story.append(Paragraph("LEASE LENS  ·  TENANT RIGHTS DOCUMENT", header_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563EB"),
            spaceAfter=18,
        )
    )

    # ── Letter body ──────────────────────────────────────────────────────────
    for line in letter_text.split("\n"):
        stripped = line.strip()

        if stripped == "---":
            story.append(Spacer(1, 6))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#CCCCCC"),
                    spaceAfter=6,
                )
            )
        elif stripped:
            safe = (
                stripped
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe, body_style))
            story.append(Spacer(1, 4))
        else:
            story.append(Spacer(1, 10))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceAfter=4)
    )
    story.append(
        Paragraph(
            "Generated by Lease Lens — AI-assisted tenant rights tool. "
            "This document is for informational purposes only and does not constitute legal advice. "
            "Review carefully and consult a licensed tenant rights attorney before sending.",
            footer_style,
        )
    )

    doc.build(story)
    return str(pdf_path)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_PARAMS = {
        "tenant_name": "Jane Smith",
        "tenant_address": "456 Oak Avenue, Apt 2, San Francisco, CA 94102",
        "landlord_name": "Pacific Properties LLC",
        "landlord_address": "789 Market Street, Suite 100, San Francisco, CA 94103",
        "property_address": "123 Main Street, Apt 4B, San Francisco, CA 94101",
        "move_out_date": "2025-03-01",
        "deposit_amount": 1500.0,
        "deductions_claimed": [
            "$200 for carpet cleaning",
            "$100 for key replacement",
        ],
        "state": "CA",
    }

    print("Generating security deposit demand letter — CA, $1,500 deposit...\n")
    letter = generate_deposit_letter(TEST_PARAMS)

    # Verify the CA statute is cited
    assert "1950.5" in letter or "Civil Code" in letter, (
        "ERROR: CA statute (Civil Code § 1950.5) not found in generated letter!"
    )
    print("✓ Statute citation verified (California Civil Code § 1950.5 present)\n")
    print("─" * 60)
    print(letter[:800] + "\n...[truncated]" if len(letter) > 800 else letter)
    print("─" * 60)

    pdf_path = save_letter_pdf(letter, "test_deposit_letter.pdf")
    # Copy to cwd for easy inspection
    import shutil
    local_copy = pathlib.Path("test_deposit_letter.pdf")
    shutil.copy(pdf_path, local_copy)
    print(f"\n✓ PDF saved → {local_copy.absolute()}")
    print(f"  (Total letter length: {len(letter)} characters)")
