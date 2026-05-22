"""
documents.py — Document generation for Lease Lens (deposits, repairs, move-out).

Public API:
  STATE_DEPOSIT_LAW            dict  — all 50 states, return deadlines & penalties
  generate_deposit_letter(params) -> str   — Ollama-drafted formal demand letter
  save_letter_pdf(letter_text, filename)   -> str (path to saved PDF)

  STATE_HABITABILITY_LAW       dict  — all 50 states, repair deadlines & tenant remedies
  generate_repair_notice(params) -> str    — Ollama-drafted formal repair demand notice
  generate_moveout_checklist(params) -> str  — PDF move-out condition checklist (path)
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
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
# Habitability / repair law — all 50 US states
# ---------------------------------------------------------------------------

STATE_HABITABILITY_LAW: dict[str, dict] = {
    "AL": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "§ 35-9A-204",
        "rent_withhold_threshold": 0.0,
    },
    "AK": {
        "repair_deadline_days": 10,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "AS 34.03.100",
        "rent_withhold_threshold": 1.0,
    },
    "AZ": {
        "repair_deadline_days": 10,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Ariz. Rev. Stat. § 33-1361",
        "rent_withhold_threshold": 1.0,
    },
    "AR": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "Ark. Code Ann. § 18-17-601",
        "rent_withhold_threshold": 0.0,
    },
    "CA": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "California Civil Code § 1942",
        "rent_withhold_threshold": 1.0,
    },
    "CO": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Colo. Rev. Stat. § 38-12-503",
        "rent_withhold_threshold": 1.0,
    },
    "CT": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "lease termination"],
        "statute_citation": "Conn. Gen. Stat. § 47a-13",
        "rent_withhold_threshold": 1.0,
    },
    "DE": {
        "repair_deadline_days": 15,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "25 Del. Code § 5305",
        "rent_withhold_threshold": 1.0,
    },
    "FL": {
        "repair_deadline_days": 7,
        "tenant_remedies": ["rent withholding", "lease termination"],
        "statute_citation": "Fla. Stat. § 83.56",
        "rent_withhold_threshold": 1.0,
    },
    "GA": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "Ga. Code Ann. § 44-7-13",
        "rent_withhold_threshold": 0.0,
    },
    "HI": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Haw. Rev. Stat. § 521-42",
        "rent_withhold_threshold": 1.0,
    },
    "ID": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "Idaho Code § 55-2004",
        "rent_withhold_threshold": 0.0,
    },
    "IL": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "765 ILCS 735/1",
        "rent_withhold_threshold": 1.0,
    },
    "IN": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "Ind. Code § 32-31-8-5",
        "rent_withhold_threshold": 0.0,
    },
    "IA": {
        "repair_deadline_days": 7,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Iowa Code § 562A.15",
        "rent_withhold_threshold": 1.0,
    },
    "KS": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Kan. Stat. Ann. § 58-2553",
        "rent_withhold_threshold": 1.0,
    },
    "KY": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Ky. Rev. Stat. Ann. § 383.635",
        "rent_withhold_threshold": 1.0,
    },
    "LA": {
        "repair_deadline_days": 15,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "La. Rev. Stat. Ann. § 9:3221",
        "rent_withhold_threshold": 0.0,
    },
    "ME": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "14 M.R.S. § 6021",
        "rent_withhold_threshold": 1.0,
    },
    "MD": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Md. Code, Real Prop. § 8-211",
        "rent_withhold_threshold": 1.0,
    },
    "MA": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Mass. Gen. Laws ch. 111 § 127L",
        "rent_withhold_threshold": 1.0,
    },
    "MI": {
        "repair_deadline_days": 7,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Mich. Comp. Laws § 125.534",
        "rent_withhold_threshold": 1.0,
    },
    "MN": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Minn. Stat. § 504B.395",
        "rent_withhold_threshold": 1.0,
    },
    "MS": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "Miss. Code Ann. § 89-8-23",
        "rent_withhold_threshold": 0.0,
    },
    "MO": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "Mo. Rev. Stat. § 441.234",
        "rent_withhold_threshold": 0.0,
    },
    "MT": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Mont. Code Ann. § 70-24-303",
        "rent_withhold_threshold": 1.0,
    },
    "NE": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Neb. Rev. Stat. § 76-1425",
        "rent_withhold_threshold": 1.0,
    },
    "NV": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Nev. Rev. Stat. § 118A.355",
        "rent_withhold_threshold": 1.0,
    },
    "NH": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.H. Rev. Stat. Ann. § 48-A:14",
        "rent_withhold_threshold": 1.0,
    },
    "NJ": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.J. Stat. Ann. § 2A:42-87",
        "rent_withhold_threshold": 1.0,
    },
    "NM": {
        "repair_deadline_days": 7,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.M. Stat. Ann. § 47-8-27",
        "rent_withhold_threshold": 1.0,
    },
    "NY": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.Y. Real Prop. Law § 235-b",
        "rent_withhold_threshold": 1.0,
    },
    "NC": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.C. Gen. Stat. § 42-42",
        "rent_withhold_threshold": 1.0,
    },
    "ND": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "N.D. Cent. Code § 47-16-13",
        "rent_withhold_threshold": 1.0,
    },
    "OH": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Ohio Rev. Code Ann. § 5321.07",
        "rent_withhold_threshold": 1.0,
    },
    "OK": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "41 Okla. Stat. § 121",
        "rent_withhold_threshold": 1.0,
    },
    "OR": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Or. Rev. Stat. § 90.365",
        "rent_withhold_threshold": 1.0,
    },
    "PA": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["rent withholding", "lease termination"],
        "statute_citation": "68 Pa. Stat. § 250.201",
        "rent_withhold_threshold": 1.0,
    },
    "RI": {
        "repair_deadline_days": 20,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "R.I. Gen. Laws § 34-18-22",
        "rent_withhold_threshold": 1.0,
    },
    "SC": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "S.C. Code Ann. § 27-40-630",
        "rent_withhold_threshold": 1.0,
    },
    "SD": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "S.D. Codified Laws § 43-32-8",
        "rent_withhold_threshold": 0.0,
    },
    "TN": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Tenn. Code Ann. § 66-28-304",
        "rent_withhold_threshold": 1.0,
    },
    "TX": {
        "repair_deadline_days": 7,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Tex. Prop. Code § 92.056",
        "rent_withhold_threshold": 1.0,
    },
    "UT": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "Utah Code Ann. § 57-22-6",
        "rent_withhold_threshold": 0.0,
    },
    "VT": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "9 Vt. Stat. Ann. § 4457",
        "rent_withhold_threshold": 1.0,
    },
    "VA": {
        "repair_deadline_days": 21,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Va. Code Ann. § 55.1-1234",
        "rent_withhold_threshold": 1.0,
    },
    "WA": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Wash. Rev. Code § 59.18.070",
        "rent_withhold_threshold": 1.0,
    },
    "WV": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["repair and deduct", "lease termination"],
        "statute_citation": "W. Va. Code § 37-6-30",
        "rent_withhold_threshold": 0.0,
    },
    "WI": {
        "repair_deadline_days": 14,
        "tenant_remedies": ["rent withholding", "repair and deduct", "lease termination"],
        "statute_citation": "Wis. Stat. § 704.07",
        "rent_withhold_threshold": 1.0,
    },
    "WY": {
        "repair_deadline_days": 30,
        "tenant_remedies": ["lease termination"],
        "statute_citation": "Wyo. Stat. Ann. § 1-21-1206",
        "rent_withhold_threshold": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Repair notice system prompt & generator
# ---------------------------------------------------------------------------

_REPAIR_NOTICE_SYSTEM_PROMPT = (
    "You are a tenant rights attorney drafting a formal repair demand notice. "
    "Write in professional, firm, legally precise language. "
    "Use standard business letter format with address blocks, salutation, numbered paragraphs, and closing. "
    "Cite the provided statute exactly as given. "
    "Do not add disclaimers, notes, or commentary — write only the notice itself."
)


def generate_repair_notice(params: dict) -> str:
    """
    Draft a formal written notice to the landlord demanding repairs,
    citing the habitability statute.

    Args:
        params: dict with keys —
            tenant_name         str
            tenant_address      str
            landlord_name       str
            landlord_address    str
            property_address    str
            issue_description   str
            date_first_reported str  (YYYY-MM-DD)
            state               str  (two-letter code)

    Returns:
        The complete notice as a plain string.
    """
    state = params.get("state", "CA").upper()
    law = STATE_HABITABILITY_LAW.get(state, STATE_HABITABILITY_LAW["CA"])

    # Parse date and compute deadline
    date_first_reported_str = str(params.get("date_first_reported", ""))
    try:
        date_first_reported_date = datetime.strptime(date_first_reported_str, "%Y-%m-%d").date()
        deadline_date = date_first_reported_date + timedelta(days=law["repair_deadline_days"])
        deadline_str = deadline_date.strftime("%B %d, %Y")
    except (ValueError, TypeError):
        deadline_str = f"{law['repair_deadline_days']} days after written notice"

    # Build numbered remedies list
    remedies_text = "\n".join(
        f"  {i + 1}. {remedy.title()}"
        for i, remedy in enumerate(law["tenant_remedies"])
    )

    today_str = date.today().strftime("%B %d, %Y")

    user_prompt = f"""
Draft a complete, formal repair demand notice using the following information.

TODAY'S DATE: {today_str}

TENANT (SENDER):
  Name: {params.get("tenant_name", "[Tenant Name]")}
  Address: {params.get("tenant_address", "[Tenant Address]")}

LANDLORD (RECIPIENT):
  Name: {params.get("landlord_name", "[Landlord Name]")}
  Address: {params.get("landlord_address", "[Landlord Address]")}

RENTAL PROPERTY: {params.get("property_address", "[Property Address]")}
DATE DEFECT FIRST REPORTED: {date_first_reported_str}
DEFECT / ISSUE DESCRIPTION: {params.get("issue_description", "[Issue Description]")}

GOVERNING STATUTE: {law["statute_citation"]}
REPAIR DEADLINE: {law["repair_deadline_days"]} days after written notice (by {deadline_str})

TENANT REMEDIES IF LANDLORD FAILS TO ACT:
{remedies_text}

The notice must include:
1. Date and address blocks (tenant address, date, landlord address).
2. Formal salutation addressing the landlord by name.
3. A clear description of the defect/condition requiring repair.
4. A statement that this constitutes WRITTEN NOTICE pursuant to {law["statute_citation"]}.
5. The exact repair deadline date: {deadline_str}.
6. The tenant's available remedies if the landlord fails to act by the deadline.
7. A request for written acknowledgement of receipt and a repair plan.
8. An instruction to send any response via certified mail.
9. A professional closing signature block.

Write the complete notice now. Output only the notice — no preamble, no notes.
""".strip()

    model = get_ollama_model()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _REPAIR_NOTICE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Move-out checklist — items, wear-and-tear definitions, PDF generator
# ---------------------------------------------------------------------------

_CHECKLIST_ITEMS: dict = {
    "common": [   # appears in both apartment and house
        ("ENTRY & COMMON AREAS", [
            "Front door — condition, locks, deadbolt, hardware",
            "Interior hallway doors",
            "Walls — holes, marks, scuffs, paint condition",
            "Ceilings — water stains, cracks, damage",
            "Flooring — scratches, stains, damage",
            "Light fixtures & ceiling fans",
            "Electrical outlets & switches — all functional",
            "Window glass — cracks, chips",
            "Window screens — tears, bent frames",
            "Window blinds / curtain rods",
            "Smoke detectors — present & functional",
            "Carbon monoxide detectors — present & functional",
        ]),
        ("KITCHEN", [
            "Refrigerator — exterior, interior, shelves, seals",
            "Stove / oven — burners, oven interior, drip pans",
            "Microwave (if landlord-provided)",
            "Dishwasher — interior, door gasket, spray arm",
            "Countertops — chips, burns, stains",
            "Cabinet doors & drawers — hinges, handles",
            "Sink basin — chips, stains",
            "Faucet — drips, function",
            "Garbage disposal — functional",
            "Backsplash tiles & grout",
            "Kitchen flooring",
        ]),
        ("BATHROOM", [
            "Toilet — flushes, no cracks, seat, tank lid",
            "Sink — basin, faucet, drain",
            "Bathtub / shower — caulk, tiles, grout, drain",
            "Shower door / curtain rod",
            "Mirror — chips, mounting",
            "Vanity cabinet",
            "Towel bars & toilet paper holder",
            "Exhaust fan — functional",
            "Flooring & wall tiles",
        ]),
        ("BEDROOM(S)", [
            "Bedroom door — function, lock",
            "Closet — door, rod, shelf",
            "Walls — holes, marks",
            "Ceiling",
            "Flooring / carpet — stains, tears, burns",
            "Windows & screens",
            "Window coverings",
            "Light fixture / ceiling fan",
        ]),
        ("LIVING / DINING AREA", [
            "Walls — marks, holes, paint",
            "Ceiling",
            "Flooring / carpet",
            "Fireplace / mantle (if present)",
            "Windows & screens",
            "Window coverings",
            "Light fixtures",
        ]),
    ],
    "house_extra": [  # appended only for property_type == "house"
        ("GARAGE", [
            "Garage door — panels, opener, remote(s)",
            "Garage interior — walls, floor",
            "Storage shelving",
            "Utility connections (if applicable)",
        ]),
        ("EXTERIOR", [
            "Front lawn — mowed, weeds, debris",
            "Back yard — mowed, weeds, debris",
            "Driveway — cracks, oil stains",
            "Walkways & steps",
            "Fencing — panels, gate hardware",
            "Outdoor lighting",
            "Hose bibs / irrigation system",
        ]),
    ],
}

_GENERIC_WEAR_TEAR = (
    "Normal wear and tear means deterioration resulting from the ordinary, reasonable use of "
    "the rental property. This includes minor scuffs, small nail holes from hanging pictures, "
    "carpet wear from foot traffic, and paint fading from sunlight. Landlords cannot charge "
    "tenants for these items."
)

_NORMAL_WEAR_TEAR: dict[str, str] = {
    "AL": _GENERIC_WEAR_TEAR,
    "AK": _GENERIC_WEAR_TEAR,
    "AZ": _GENERIC_WEAR_TEAR,
    "AR": _GENERIC_WEAR_TEAR,
    "CA": (
        "Under California Civil Code § 1950.5(e), normal wear and tear means deterioration "
        "which occurs through ordinary use of the premises. The landlord cannot charge for "
        "paint fading, minor scuffs, or carpet wear from normal foot traffic."
    ),
    "CO": _GENERIC_WEAR_TEAR,
    "CT": _GENERIC_WEAR_TEAR,
    "DE": _GENERIC_WEAR_TEAR,
    "FL": (
        "Under Fla. Stat. § 83.43, normal wear and tear means deterioration from reasonable "
        "use. Landlords cannot deduct for repainted walls, minor scuffs, or normal carpet wear."
    ),
    "GA": _GENERIC_WEAR_TEAR,
    "HI": _GENERIC_WEAR_TEAR,
    "ID": _GENERIC_WEAR_TEAR,
    "IL": (
        "Under the Chicago Residential Landlord Ordinance and 765 ILCS, normal wear and tear "
        "includes minor scuffs, small nail holes, and faded paint that result from ordinary "
        "habitation."
    ),
    "IN": _GENERIC_WEAR_TEAR,
    "IA": _GENERIC_WEAR_TEAR,
    "KS": _GENERIC_WEAR_TEAR,
    "KY": _GENERIC_WEAR_TEAR,
    "LA": _GENERIC_WEAR_TEAR,
    "ME": _GENERIC_WEAR_TEAR,
    "MD": _GENERIC_WEAR_TEAR,
    "MA": (
        "Under Mass. Gen. Laws ch. 186 § 15B, landlords may only deduct for damage beyond "
        "normal wear and tear, defined as deterioration from reasonable use."
    ),
    "MI": _GENERIC_WEAR_TEAR,
    "MN": _GENERIC_WEAR_TEAR,
    "MS": _GENERIC_WEAR_TEAR,
    "MO": _GENERIC_WEAR_TEAR,
    "MT": _GENERIC_WEAR_TEAR,
    "NE": _GENERIC_WEAR_TEAR,
    "NV": _GENERIC_WEAR_TEAR,
    "NH": _GENERIC_WEAR_TEAR,
    "NJ": _GENERIC_WEAR_TEAR,
    "NM": _GENERIC_WEAR_TEAR,
    "NY": (
        "Under N.Y. Real Prop. Law, normal wear and tear includes minor scuffs, faded paint, "
        "and light carpet wear from regular use. Landlords cannot charge for these items."
    ),
    "NC": _GENERIC_WEAR_TEAR,
    "ND": _GENERIC_WEAR_TEAR,
    "OH": _GENERIC_WEAR_TEAR,
    "OK": _GENERIC_WEAR_TEAR,
    "OR": _GENERIC_WEAR_TEAR,
    "PA": _GENERIC_WEAR_TEAR,
    "RI": _GENERIC_WEAR_TEAR,
    "SC": _GENERIC_WEAR_TEAR,
    "SD": _GENERIC_WEAR_TEAR,
    "TN": _GENERIC_WEAR_TEAR,
    "TX": (
        "Under Tex. Prop. Code § 92.101, normal wear and tear means deterioration that results "
        "from intended use. Examples include minor nail holes, carpet worn from regular use, "
        "and faded paint."
    ),
    "UT": _GENERIC_WEAR_TEAR,
    "VT": _GENERIC_WEAR_TEAR,
    "VA": _GENERIC_WEAR_TEAR,
    "WA": (
        "Under Wash. Rev. Code § 59.18.285, normal wear and tear means deterioration that "
        "occurs from the intended use of the rental unit. Small nail holes and light carpet "
        "wear are examples."
    ),
    "WV": _GENERIC_WEAR_TEAR,
    "WI": _GENERIC_WEAR_TEAR,
    "WY": _GENERIC_WEAR_TEAR,
}


def generate_moveout_checklist(params: dict) -> str:
    """
    Generate a move-out condition checklist PDF (no LLM).

    Args:
        params: dict with keys —
            state           str  (two-letter code)
            move_in_date    str  (YYYY-MM-DD)
            property_type   str  ("apartment" or "house")
            tenant_name     str  (optional)
            property_address str (optional)

    Returns:
        Absolute path to the saved PDF file.
    """
    property_type = params.get("property_type", "apartment").lower()

    sections = list(_CHECKLIST_ITEMS["common"])
    if property_type == "house":
        sections += list(_CHECKLIST_ITEMS["house_extra"])

    state = params.get("state", "CA").upper()
    wear_tear = _NORMAL_WEAR_TEAR.get(state, _NORMAL_WEAR_TEAR["CA"])

    return _save_checklist_pdf(params, sections, wear_tear)


def _save_checklist_pdf(params: dict, sections: list, wear_tear_note: str) -> str:
    """
    Render the move-out checklist as a PDF using ReportLab.

    Returns:
        Absolute path to the saved PDF.
    """
    state = params.get("state", "CA").upper()
    move_in_date = params.get("move_in_date", "")
    property_type = params.get("property_type", "apartment").title()
    tenant_name = params.get("tenant_name", "")
    property_address = params.get("property_address", "")

    output_dir = pathlib.Path(tempfile.mkdtemp())
    pdf_path = output_dir / "moveout_checklist.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
        title="Move-Out Condition Checklist",
        author="Lease Lens",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CLTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "CLSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    meta_style = ParagraphStyle(
        "CLMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    wt_header_style = ParagraphStyle(
        "CLWTHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#1E40AF"),
        spaceAfter=2,
    )
    wt_body_style = ParagraphStyle(
        "CLWTBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#1E3A8A"),
        leading=13,
        spaceAfter=0,
    )
    footer_style = ParagraphStyle(
        "CLFooter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
        spaceBefore=10,
    )

    story: list = []

    # ── Title block ──────────────────────────────────────────────────────────
    story.append(Paragraph("MOVE-OUT CONDITION CHECKLIST", title_style))

    subtitle_parts = []
    if tenant_name:
        subtitle_parts.append(f"Tenant: {tenant_name}")
    if property_address:
        subtitle_parts.append(f"Property: {property_address}")
    if subtitle_parts:
        story.append(Paragraph("  |  ".join(subtitle_parts), subtitle_style))

    story.append(
        Paragraph(
            f"State: {state}  |  Move-in: {move_in_date}  |  Property type: {property_type}",
            meta_style,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563EB"),
            spaceAfter=8,
        )
    )

    # ── Wear & Tear box ──────────────────────────────────────────────────────
    wt_table_data = [
        [
            Paragraph(f"NORMAL WEAR &amp; TEAR — {state}", wt_header_style),
        ],
        [
            Paragraph(wear_tear_note, wt_body_style),
        ],
    ]
    available_width = LETTER[0] - 1.5 * inch  # left + right margins
    wt_table = Table(wt_table_data, colWidths=[available_width])
    wt_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor("#2563EB")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ])
    )
    story.append(wt_table)
    story.append(Spacer(1, 12))

    # ── Checklist sections ───────────────────────────────────────────────────
    item_style_base = ParagraphStyle(
        "CLItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )
    label_style = ParagraphStyle(
        "CLLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#6B7280"),
        leading=12,
    )

    col_widths = [available_width * 0.60, available_width * 0.25, available_width * 0.15]

    for section_name, items in sections:
        # Section header row
        header_para = Paragraph(
            f'<font color="white"><b>{section_name}</b></font>',
            ParagraphStyle(
                "CLSectionHeader",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=10,
                textColor=colors.white,
                leading=14,
            ),
        )
        section_header_table = Table([[header_para, "", ""]], colWidths=col_widths)
        section_header_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2563EB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("SPAN", (0, 0), (-1, -1)),
            ])
        )
        story.append(section_header_table)

        # Item rows
        table_data = []
        for i, item_text in enumerate(items):
            row_bg = colors.HexColor("#F8FAFC") if i % 2 == 0 else colors.white
            item_para = Paragraph(f"☐  {item_text}", item_style_base)
            cond_para = Paragraph("Condition: _______________", label_style)
            photo_para = Paragraph("Photo #____", label_style)
            table_data.append([item_para, cond_para, photo_para])

        if table_data:
            items_table = Table(table_data, colWidths=col_widths)
            style_commands = [
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#E2E8F0")),
            ]
            # Alternate row shading
            for i in range(len(table_data)):
                if i % 2 == 0:
                    style_commands.append(
                        ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC"))
                    )
                else:
                    style_commands.append(
                        ("BACKGROUND", (0, i), (-1, i), colors.white)
                    )
            items_table.setStyle(TableStyle(style_commands))
            story.append(items_table)

        story.append(Spacer(1, 8))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceAfter=4)
    )
    story.append(
        Paragraph(
            "Generated by Lease Lens  ·  Informational only — not legal advice",
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
