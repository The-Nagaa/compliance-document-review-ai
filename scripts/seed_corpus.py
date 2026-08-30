"""
Corpus Seeding Script for Compliance Document Review App.

Generates realistic synthetic datasets:
1. 35 Compliance Rules (prohibited claims, performance standards, disclosures, supervision)
2. 25 Standard Disclosures (SEC, FINRA, SIPC, tax, risk, testimonials)
3. 100 Reviewed Precedent Documents (with decisions: approved/rejected/needs_revision and officer comments)

Zero Real Client Data. Safe for all environments.
"""

import sys
from pathlib import Path

# Ensure root workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai.app.core.config import settings
from ai.app.core.logging import logger
from ai.app.models.entities import Disclosure, Precedent, Rule
from ai.app.repositories.vector_store import vector_store


# --- 1. COMPLIANCE RULES CORPUS (35 Rules) ---
RULES_DATA = [
    # Category: Prohibited Claims & Guarantees
    ("RULE-001", "Prohibition of Guaranteed Returns: Communications must never state or imply that any investment return or profit is guaranteed, assured, or risk-free.", "PROHIBITED_CLAIMS"),
    ("RULE-002", "Prohibition of Exaggerated or Misleading Claims: Statements must not exaggerate the benefits of an investment or use sensationalist phrases such as 'can't lose' or 'foolproof'.", "PROHIBITED_CLAIMS"),
    ("RULE-003", "Protection Against Principal Loss Disclaimers: It is strictly prohibited to claim that principal is protected or insured unless specifically backed by FDIC/NCUA insurance with explicit details.", "PROHIBITED_CLAIMS"),
    ("RULE-004", "Elimination of Downside Risk: Marketing materials cannot promise complete downside protection or zero risk in equity or alternative investments.", "PROHIBITED_CLAIMS"),
    ("RULE-005", "Absolute Safety Assertions: Promising that funds are '100% safe' or immune to market volatility is a severe regulatory violation.", "PROHIBITED_CLAIMS"),

    # Category: Performance Claims & Standards
    ("RULE-010", "Past Performance Disclaimer: Any reference to past investment performance or historical returns must be accompanied by the statement 'Past performance is no guarantee of future results.'", "PERFORMANCE_CLAIMS"),
    ("RULE-011", "Net of Fees Performance Presentation: Performance figures must be presented net of all advisory fees and brokerage commissions, or alongside clear gross-to-net fee deduction disclosures.", "PERFORMANCE_CLAIMS"),
    ("RULE-012", "Hypothetical and Backtested Performance: Hypothetical, model, or backtested performance must be explicitly labeled as such with full methodology disclosure and risk caveats.", "PERFORMANCE_CLAIMS"),
    ("RULE-013", "Cherry-Picking Timeframes: Presenting performance over selective, non-standard time periods to artificially inflate track record is prohibited. Must include standard 1, 5, and 10-year periods if available.", "PERFORMANCE_CLAIMS"),
    ("RULE-014", "Benchmark Comparison Standards: Benchmark comparisons must use relevant, broad-based indices reflecting the asset class and strategy, and disclose index composition limitations.", "PERFORMANCE_CLAIMS"),
    ("RULE-015", "Unsubstantiated Return Targets: Stating specific forward-looking percentage return targets without detailed risk scenarios and probability models is prohibited.", "PERFORMANCE_CLAIMS"),
    ("RULE-016", "Compound Annual Growth Rate (CAGR) Accuracy: Any calculated CAGR or annual rate of return must explicitly disclose start and end calculation dates.", "PERFORMANCE_CLAIMS"),

    # Category: Required Disclosures & Disclaimers
    ("RULE-020", "SEC Registered Investment Advisor Disclosure: Written communications from an RIA must disclose: 'Advisory services offered through [Firm Name], an SEC Registered Investment Advisor.'", "REQUIRED_DISCLOSURES"),
    ("RULE-021", "Risk of Loss Disclosure: All investment presentations must contain the prominent disclosure: 'Investing in securities involves risk of loss that clients should be prepared to bear.'", "REQUIRED_DISCLOSURES"),
    ("RULE-022", "SIPC / FINRA Membership Disclosure: Broker-dealer communications must state 'Member FINRA / SIPC' when referencing brokerage accounts or clearing services.", "REQUIRED_DISCLOSURES"),
    ("RULE-023", "Tax and Legal Advice Disclaimer: Financial planning materials must include a disclaimer that the firm does not provide legal or tax advice, and clients should consult their tax advisor.", "REQUIRED_DISCLOSURES"),
    ("RULE-024", "Fee Schedule and Billing Disclosure: When discussing investment advisory fees, materials must disclose where the client can obtain the full Form ADV Part 2A brochure.", "REQUIRED_DISCLOSURES"),
    ("RULE-025", "Material Conflict of Interest Disclosure: Any economic incentive or referral fee received for recommending specific investment products or custodians must be clearly disclosed.", "REQUIRED_DISCLOSURES"),
    ("RULE-026", "Third-Party Ranking and Award Disclosures: Any mention of rankings, awards, or 'Top Advisor' recognitions must disclose the granting entity, criteria, and whether any compensation was paid.", "REQUIRED_DISCLOSURES"),
    ("RULE-027", "Affiliated Custodian Notice: Clear notice must be provided when assets are held with an affiliated custodian or brokerage clearing firm.", "REQUIRED_DISCLOSURES"),

    # Category: Testimonials, Endorsements & Social Media
    ("RULE-030", "Testimonial Disclosure under SEC Marketing Rule: Testimonials from current clients must clearly disclose that the statement was made by a client and whether compensation was provided.", "TESTIMONIALS"),
    ("RULE-031", "Endorsement by Non-Clients: Endorsements by third parties or influencers must disclose that the person is not a current client and note any material conflicts of interest.", "TESTIMONIALS"),
    ("RULE-032", "Social Media Interactive Communications: Public social media posts discussing investment strategies must be archived and reviewed in accordance with firm supervision policies.", "SUPERVISION"),
    ("RULE-033", "Promissory Client Quotes: Quoting a client stating they 'made millions' without balanced risk representation is misleading and prohibited.", "TESTIMONIALS"),

    # Category: Supervisory & Approval Standards
    ("RULE-040", "Supervisory Principal Pre-Approval: All mass-distributed client marketing materials, brochures, and email blasts must receive written approval from a registered compliance principal prior to distribution.", "SUPERVISION"),
    ("RULE-041", "Material Revision Tracking: Any substantial revision to an approved marketing piece requires re-submission and re-approval before use.", "SUPERVISION"),
    ("RULE-042", "Advisor Name and CRD Identification: Communications must clearly identify the individual advisor's name and firm affiliation.", "SUPERVISION"),
    ("RULE-043", "Cryptocurrency and Digital Asset Risk Warning: Communications discussing digital assets or cryptocurrency funds must contain explicit warnings regarding volatility and lack of SIPC protection.", "REQUIRED_DISCLOSURES"),
    ("RULE-044", "Private Placement / Regulation D Warnings: Materials referencing private funds must state they are offered exclusively to Accredited Investors with significant illiquidity risk.", "REQUIRED_DISCLOSURES"),
    ("RULE-045", "Option and Margin Trading Risk Disclosures: Discussions involving options or margin trading must include references to the Characteristics and Risks of Standardized Options.", "REQUIRED_DISCLOSURES"),
    ("RULE-046", "Annuity and Insurance Guarantee Limitations: Any discussion of insurance or annuity guarantees must specify that guarantees are subject to the claims-paying ability of the issuing insurer.", "REQUIRED_DISCLOSURES"),
    ("RULE-047", "Structured Notes and Derivative Risks: Marketing structured notes must disclose credit risk of the issuer, cap rates, barrier levels, and secondary market liquidity limitations.", "REQUIRED_DISCLOSURES"),
    ("RULE-048", "Sustainable / ESG Investment Objective Substantiation: Claims regarding ESG or sustainable criteria must define the specific screening methodology and benchmarks used.", "PROHIBITED_CLAIMS"),
    ("RULE-049", "Prompt Delivery of Privacy Policy: Advisor correspondence must state how clients can view the firm's Regulation S-P privacy notice.", "REQUIRED_DISCLOSURES"),
]


# --- 2. STANDARD DISCLOSURES CORPUS (25 Disclosures) ---
DISCLOSURES_DATA = [
    ("DISC-001", "Advisory services offered through Beacon Wealth Management LLC, an SEC Registered Investment Advisor.", "SEC_RIA_DISCLOSURE"),
    ("DISC-002", "Past performance is no guarantee of future results. Investments are subject to market risk, including possible loss of the principal amount invested.", "PAST_PERFORMANCE"),
    ("DISC-003", "Securities offered through Horizon Financial Services, Member FINRA/SIPC. Investment advisory products are not FDIC insured.", "FINRA_SIPC_MEMBERSHIP"),
    ("DISC-004", "Beacon Wealth Management LLC does not provide tax, legal, or accounting advice. Please consult your individual tax professional regarding your personal situation.", "TAX_LEGAL_DISCLAIMER"),
    ("DISC-005", "Hypothetical performance results have many inherent limitations and do not represent actual trading. No representation is being made that any account will achieve profits or losses similar to those shown.", "HYPOTHETICAL_PERFORMANCE"),
    ("DISC-006", "All investing involves risk, including the potential loss of principal. There can be no assurance that any investment strategy will be successful or profitable.", "GENERAL_RISK_OF_LOSS"),
    ("DISC-007", "For complete information regarding advisory fees, billing practices, and conflicts of interest, please review our Form ADV Part 2A Brochure available at advisorinfo.sec.gov.", "FORM_ADV_BROCHURE"),
    ("DISC-008", "Rankings, awards, and recognitions by unaffiliated publications are based on quantitative and qualitative criteria and should not be construed as an endorsement of the advisor.", "THIRD_PARTY_RANKING"),
    ("DISC-009", "Testimonials and endorsements may not be representative of the experience of other clients and do not guarantee future investment performance or satisfaction.", "TESTIMONIAL_DISCLAIMER"),
    ("DISC-010", "Digital assets and cryptocurrencies are highly volatile, speculative, and not protected by SIPC or FDIC insurance.", "DIGITAL_ASSET_RISK"),
    ("DISC-011", "Alternative investments and private placements are illiquid, speculative, and suitable only for accredited or qualified institutional investors.", "PRIVATE_PLACEMENT_ACCREDITED"),
    ("DISC-012", "Options trading involves substantial risk and is not suitable for all investors. Prior to buying or selling options, read the Characteristics and Risks of Standardized Options.", "OPTIONS_DISCLOSURE"),
    ("DISC-013", "Insurance and annuity product guarantees are backed solely by the financial strength and claims-paying ability of the issuing insurance company.", "INSURANCE_GUARANTEE"),
    ("DISC-014", "Municipal bond interest may be subject to the alternative minimum tax (AMT) and state or local taxes depending on the investor's residency.", "MUNICIPAL_TAX_DISCLAIMER"),
    ("DISC-015", "International and emerging market investments involve currency exchange fluctuations, political instability, and differing accounting standards.", "INTERNATIONAL_INVESTING"),
    ("DISC-016", "Indices are unmanaged and cannot be invested in directly. Index performance does not reflect the deduction of management fees, transaction costs, or expenses.", "BENCHMARK_UNMANAGED_INDEX"),
    ("DISC-017", "Fixed income securities are subject to interest rate risk, credit risk, inflation risk, and reinvestment risk.", "FIXED_INCOME_RISK"),
    ("DISC-018", "Margin trading carries a high level of risk and may result in losses exceeding the initial deposit. Please review our Margin Disclosure Statement.", "MARGIN_RISK"),
    ("DISC-019", "Real estate investments are subject to fluctuations in property values, tenancy risks, leverage, and economic downturns.", "REAL_ESTATE_RISK"),
    ("DISC-020", "Fee deductions reduce overall portfolio returns. Net returns shown reflect the maximum annual management fee of 1.00% unless specified otherwise.", "FEE_IMPACT_DISCLOSURE"),
    ("DISC-021", "ESG and sustainable investing strategies may limit the universe of available investments and may not align with traditional benchmark allocations.", "ESG_DISCLOSURE"),
    ("DISC-022", "Structured notes are complex debt securities subject to credit risk of the issuing financial institution and may experience principal loss.", "STRUCTURED_NOTES"),
    ("DISC-023", "Our firm's privacy policy explaining how we safeguard non-public personal information under Regulation S-P is available upon request.", "PRIVACY_POLICY_REG_SP"),
    ("DISC-024", "Performance calculations presented gross of fees do not reflect advisory fee deductions, which compound over time to reduce net returns.", "GROSS_OF_FEES_WARNING"),
    ("DISC-025", "Material contained herein is for informational and educational purposes only and does not constitute an offer to buy or sell securities.", "GENERAL_INFORMATIONAL_ONLY"),
]


# --- 3. SYNTHETIC REVIEWED PRECEDENTS CORPUS (100 Documents) ---
def generate_synthetic_precedents() -> list[Precedent]:
    """Generates 100 realistic, reviewed historical submissions with decisions and compliance officer feedback."""
    precedents: list[Precedent] = []

    templates = [
        # Set 1: Approved documents (clean marketing & quarterly updates)
        (
            "approved",
            "Quarterly Market Commentary prepared for [CLIENT_1]. We reviewed economic trends, inflation indicators, and equity market valuations across diversified index holdings. Advisory services offered through Beacon Wealth Management LLC, an SEC Registered Investment Advisor. Past performance is no guarantee of future results. All investing involves risk including potential loss of principal.",
            "Fully compliant. Required SEC registration disclosure and past performance disclaimers are prominently displayed. Balanced economic commentary with no prohibited claims."
        ),
        (
            "approved",
            "Educational workshop presentation on Retirement Income Planning. Covers social security optimization, tax-bracket management, and asset allocation across fixed income and equities. Beacon Wealth Management LLC does not provide tax advice. Securities offered through Horizon Financial Services, Member FINRA/SIPC.",
            "Approved without changes. Clear tax disclaimer present, balanced risk discussion, and correct FINRA/SIPC membership attribution."
        ),
        (
            "approved",
            "Client proposal letter for [CLIENT_1] outlining balanced 60/40 asset allocation strategy. Estimated expense ratios provided net of advisory fees. Complete Form ADV Part 2A Brochure reference included. Past performance disclaimer attached to historical index charts.",
            "Approved. Proper net-of-fees fee breakdown and required disclosures attached to index performance charts."
        ),
        (
            "approved",
            "Email newsletter discussing municipal bond yield strategies. Highlights tax-equivalent yield calculation for high-net-worth investors. Discloses municipal bond AMT risk and state tax nuances. Discloses firm RIA status.",
            "Approved. Sound educational content with required municipal tax caveats and firm affiliation disclosure."
        ),

        # Set 2: Rejected documents (guaranteed claims, misleading returns, missing required elements)
        (
            "rejected",
            "Marketing email to prospective clients promising guaranteed 18% annual return on private tech fund portfolio. Claims investments are '100% risk-free and protected against stock market crashes'. Contact [EMAIL_1] to lock in your guaranteed spot.",
            "REJECTED. Flagrant violation of RULE-001 (Prohibition of Guaranteed Returns) and RULE-002 (Misleading Claims). Falsely claims risk-free principal protection. Must be completely removed from circulation."
        ),
        (
            "rejected",
            "Social media ad: 'Beat Wall Street every year! Our proprietary algorithmic strategy produced 45% gains last quarter with zero losing trades. Never worry about retirement again.' No disclosures or firm registration provided.",
            "REJECTED. Prohibited promissory language, unapproved cherry-picked quarterly performance, and complete omission of SEC RIA and risk disclosures (RULE-002, RULE-013, RULE-020)."
        ),
        (
            "rejected",
            "Client flyer for [CLIENT_1] stating: 'Guaranteed principal growth backed by our firm's private reserve fund. Safe high yield with zero downside volatility.'",
            "REJECTED. Violation of RULE-001, RULE-003, and RULE-005. Promising zero downside risk and unapproved private guarantees is strictly illegal under compliance standards."
        ),
        (
            "rejected",
            "Promotional brochure featuring unvetted client quotes claiming our firm doubled their money in six months without risk. Fails to disclose client status or compensation under the SEC Marketing Rule.",
            "REJECTED. SEC Marketing Rule violation (RULE-030). Misleading testimonials with unsubstantiated promissory claims."
        ),

        # Set 3: Needs Revision (missing standard disclosures, fixable phrasing, gross-of-fees)
        (
            "needs_revision",
            "Quarterly performance report for account [ACCOUNT_1]. Shows 3-year historical returns against S&P 500 benchmark. However, returns are presented gross of management fees, and the past performance disclaimer is missing from page 2.",
            "Needs Revision. Please add the standard 'Past performance is no guarantee of future results' disclaimer (RULE-010) and include the net-of-fees return schedule (RULE-011) before distribution."
        ),
        (
            "needs_revision",
            "Brochure introducing our Sustainable Energy Strategy. Discusses ESG screening methodology and anticipated 12% target return, but lacks the SEC RIA disclosure statement and general risk of principal loss disclaimer.",
            "Needs Revision. Tone is acceptable, but mandatory RIA registration notice (RULE-020) and Risk of Loss disclaimer (RULE-021) must be inserted in the footer."
        ),
        (
            "needs_revision",
            "Email draft to [CLIENT_1] discussing high-yield corporate debt strategy. Compares yield to bank CDs without clarifying that corporate bonds are not FDIC insured and carry default risk.",
            "Needs Revision. Must clarify the distinction between bank deposits and corporate fixed income, and add the fixed income credit risk disclosure (RULE-003, DISC-017)."
        ),
        (
            "needs_revision",
            "Presentation on Options Hedging for executive stock compensation. Details protective put strategies but omits reference to the Characteristics and Risks of Standardized Options booklet.",
            "Needs Revision. Required options disclosure statement (DISC-012) must be appended prior to client delivery."
        ),
    ]

    topics = [
        "Tech Growth Equity", "Fixed Income Core", "Municipal Bond Portfolio",
        "Retirement Income Strategy", "Tax-Loss Harvesting Guide", "Estate Planning Overview",
        "Dividend Aristocrats Strategy", "Global Emerging Markets", "Small Cap Opportunities",
        "Real Estate Income Trust", "Healthcare Innovations Fund", "Green Energy Transition",
        "Treasury Inflation-Protected Securities", "Conservative Balanced Portfolio",
        "High Yield Corporate Credit", "Private Credit Overview", "Structured Note Strategy",
        "Multi-Asset Preservation Fund", "529 College Savings Guide", "Executive Equity Compensation"
    ]

    # Generate 100 distinct precedent records across permutations
    doc_index = 101
    for i in range(100):
        t_decision, t_text, t_comment = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        client_tag = f"[CLIENT_{(i % 5) + 1}]"
        acct_tag = f"[ACCOUNT_{(i % 8) + 1}]"

        varied_text = (
            f"Document Title: {topic} Review for {client_tag} (Ref: {acct_tag}).\n"
            f"Prepared by Senior Advisor for compliance review.\n"
            f"{t_text}\n"
            f"Asset Classification: {topic}. Target Allocation: Diversified Holdings."
        )

        doc_id = f"PREC-DOC-{doc_index + i:04d}"
        precedents.append(
            Precedent(
                document_id=doc_id,
                masked_text=varied_text,
                decision=t_decision,
                comment=t_comment,
                metadata={"topic": topic, "index": i + 1}
            )
        )

    return precedents


def seed_all(persist: bool = True) -> None:
    """Populate and persist all seed data into vector store."""
    logger.info("Starting synthetic compliance corpus seeding...")

    # 1. Seed Rules
    rules = [
        Rule(id=r_id, text=r_text, category=r_cat)
        for r_id, r_text, r_cat in RULES_DATA
    ]
    vector_store.add_rules_batch(rules)
    logger.info(f"Seeded {len(rules)} compliance rules.")

    # 2. Seed Disclosures
    disclosures = [
        Disclosure(id=d_id, text=d_text, type=d_type)
        for d_id, d_text, d_type in DISCLOSURES_DATA
    ]
    vector_store.add_disclosures_batch(disclosures)
    logger.info(f"Seeded {len(disclosures)} mandatory disclosures.")

    # 3. Seed Precedents
    precedents = generate_synthetic_precedents()
    vector_store.add_precedents_batch(precedents)
    logger.info(f"Seeded {len(precedents)} historically reviewed precedent documents.")

    if persist:
        vector_store.save_to_disk(settings.VECTOR_STORE_PATH)
        logger.info(f"Corpus successfully written to disk at {settings.VECTOR_STORE_PATH}")


if __name__ == "__main__":
    seed_all(persist=True)
