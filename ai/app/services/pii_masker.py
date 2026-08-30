"""
Server-Side PII Masking Service.

Complies with the privacy specification:
1. Replaces sensitive entities with stable, deterministic placeholders ([CLIENT_1], [EMAIL_1], etc.)
2. Preserves consistent mappings across repeated occurrences in the same document.
3. Keeps the PII mapping strictly server-side.
4. Provides unmask_text() exclusively for authorized UI display.

Known Limitations:
- Single-word names without titles/honorifics or structural cues may not be detected to avoid high false positives on common financial vocabulary.
- Non-standard international address formats without city/state/zip indicators or address prefixes may be missed.
- Deliberately obfuscated PII (e.g., 'john at example dot com') requires semantic NER rather than regex/heuristics.
"""

import re
from typing import Dict, List, Tuple
from ai.app.models.entities import PIIMapping


class PIIMasker:
    """
    Regex and heuristic-based PII Masker with stable placeholder allocation.
    """

    # 1. Emails
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    )

    # 2. SSN / Tax ID (e.g., 123-45-6789 or SSN: 123456789)
    SSN_PATTERN = re.compile(
        r'\b(?:\d{3}-\d{2}-\d{4}|(?:SSN|TIN|Tax\s*ID)[:\s#]*\d{9})\b',
        re.IGNORECASE
    )

    # 3. Account Numbers (with prefix context: Account #12345678, Acct 98765432, Account Number: 123456789012)
    ACCOUNT_PATTERN = re.compile(
        r'\b(?:Account|Acct|Portfolio|Folio|Policy|Custody|Ref)(?:\s+(?:Number|No|Num|#|ID|Code))?[\s#.:\-]+([A-Za-z0-9\-]{5,25})\b',
        re.IGNORECASE
    )

    # 4. Phone Numbers (US, international formats, and contiguous 10-digit numbers)
    PHONE_PATTERN = re.compile(
        r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?|\b\d{3}[-.\s])\d{3}[-.\s]?\d{4}\b|\b(?:Phone|Mobile|Tel|Cell|Contact)[: \t]*(\+?\d{1,3}[-.\s]?)?(\d{10})\b)',
        re.IGNORECASE
    )

    # 5. Addresses (Standard street suffix or explicit address prefix on the same line)
    ADDRESS_PREFIX_PATTERN = re.compile(
        r'\b(?:Address|Residential Address|Office Address|Location|Mailing Address)[: \t]+([^\r\n]{5,80})',
        re.IGNORECASE
    )
    ADDRESS_PATTERN = re.compile(
        r'\b\d{1,5}[ \t]+[A-Za-z0-9 \t.,]{2,40}?[ \t]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Plaza|Plz|Suite|Ste|Apt|Unit|Terrace|Ter|Place|Pl|Circle|Cir|Parkway|Pkwy|Trail|Trl|Highway|Hwy|Sector|Block|Nagar|Colony|Marg)\b'
        r'(?:[ \t]*,[ \t]*[A-Za-z \t]{2,25})*(?:[ \t]*,[ \t]*[A-Z]{2})?(?:[ \t]*,?[ \t]*\d{4,6}(?:-\d{4})?)?',
        re.IGNORECASE
    )

    # 6. Dollar amounts tied to named individuals or specific investment/account prompts
    AMOUNT_TIED_PATTERN = re.compile(
        r'(?:(?:portfolio|account|balance|investment|net\s*worth|assets|deposit)\s*(?:of|for|belonging to|value:?)\s*|\b(?:for|of|value:?)\s+|\b(?:Investment\s+Amount|Amount|Deposit|Portfolio\s+Value)[: \t]+)'
        r'(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars))',
        re.IGNORECASE
    )

    # 7. Name heuristics (Titles, Client / Client Name: markers, Dear markers, Full Name pairs)
    NAME_PREFIX_PATTERN = re.compile(
        r'\b(?:Client(?:\s+Name)?|Customer(?:\s+Name)?|Advisor(?:\s+Name)?|Investor(?:\s+Name)?|Owner(?:\s+Name)?|Beneficiary(?:\s+Name)?|Account\s+Holder|Attn|Attention|Dear|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)[: \t]+'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})\b'
    )

    # Common words / financial terms that should NEVER be masked as names
    FALSE_POSITIVE_NAME_EXCLUSIONS = {
        "Compliance Officer", "Financial Advisor", "Chief Executive", "Managing Director",
        "Mutual Fund", "Index Fund", "Stock Market", "Federal Reserve", "Treasury Bond",
        "United States", "Wall Street", "New York", "San Francisco", "Registered Investment",
        "Investment Advisor", "Terms Conditions", "Important Notice", "Privacy Policy",
        "Past Performance", "Capital Appreciation", "High Yield", "Real Estate"
    }

    def __init__(self):
        pass

    def mask_text(
        self, text: str, document_id: str = "doc_default"
    ) -> Tuple[str, Dict[str, str], List[PIIMapping]]:
        """
        Mask PII entities in raw text with stable placeholders.

        Returns:
            masked_text: Text safe for external transmission
            mapping: Dict of {placeholder: original_value}
            pii_records: List of PIIMapping entities for database storage
        """
        if not text:
            return "", {}, []

        placeholder_to_original: Dict[str, str] = {}
        original_to_placeholder: Dict[str, str] = {}
        pii_records: List[PIIMapping] = []

        counters = {
            "CLIENT": 0,
            "EMAIL": 0,
            "PHONE": 0,
            "ADDRESS": 0,
            "ACCOUNT": 0,
            "SSN": 0,
            "AMOUNT": 0,
        }

        def get_or_create_placeholder(original: str, entity_type: str) -> str:
            clean_orig = original.strip()
            if not clean_orig:
                return ""
            if clean_orig in original_to_placeholder:
                return original_to_placeholder[clean_orig]

            counters[entity_type] += 1
            placeholder = f"[{entity_type}_{counters[entity_type]}]"
            original_to_placeholder[clean_orig] = placeholder
            placeholder_to_original[placeholder] = clean_orig

            pii_records.append(
                PIIMapping(
                    document_id=document_id,
                    placeholder=placeholder,
                    original_value=clean_orig,
                    entity_type=entity_type,
                )
            )
            return placeholder

        working_text = text

        # 1. Mask Emails (high confidence)
        for match in self.EMAIL_PATTERN.finditer(working_text):
            orig = match.group(0)
            get_or_create_placeholder(orig, "EMAIL")

        # 2. Mask SSN / Tax IDs
        for match in self.SSN_PATTERN.finditer(working_text):
            orig = match.group(0)
            get_or_create_placeholder(orig, "SSN")

        # 3. Mask Account Numbers (extract the identifier group)
        for match in self.ACCOUNT_PATTERN.finditer(working_text):
            acct_val = match.group(1).strip()
            get_or_create_placeholder(acct_val, "ACCOUNT")

        # 4. Mask Contextual Names (e.g. "Client Name: John Smith" or "Dear Alice Walker")
        for match in self.NAME_PREFIX_PATTERN.finditer(working_text):
            name = match.group(1).strip()
            if name not in self.FALSE_POSITIVE_NAME_EXCLUSIONS:
                get_or_create_placeholder(name, "CLIENT")

        # 5. Mask Addresses
        for match in self.ADDRESS_PREFIX_PATTERN.finditer(working_text):
            orig = match.group(1).strip()
            get_or_create_placeholder(orig, "ADDRESS")

        for match in self.ADDRESS_PATTERN.finditer(working_text):
            orig = match.group(0).strip()
            get_or_create_placeholder(orig, "ADDRESS")

        # 6. Mask Contextual Dollar Amounts
        for match in self.AMOUNT_TIED_PATTERN.finditer(working_text):
            amount_val = match.group(1).strip()
            get_or_create_placeholder(amount_val, "AMOUNT")

        # 7. Mask Phone Numbers
        for match in self.PHONE_PATTERN.finditer(working_text):
            full_match = match.group(0)
            # If match has groups (e.g. from Phone: 9876543210)
            phone_num = match.group(2) if len(match.groups()) >= 2 and match.group(2) else full_match
            # Avoid matching single numbers or accounts already handled
            digits = re.sub(r'\D', '', phone_num)
            if len(digits) >= 10 and phone_num.strip() not in original_to_placeholder:
                get_or_create_placeholder(phone_num.strip(), "PHONE")

        # Replace all identified entities in text (longest original strings first to avoid sub-string collisions)
        sorted_originals = sorted(
            original_to_placeholder.keys(), key=lambda s: len(s), reverse=True
        )

        for orig in sorted_originals:
            placeholder = original_to_placeholder[orig]
            # Exact regex boundary or escaped replacement
            pattern = re.compile(re.escape(orig))
            working_text = pattern.sub(placeholder, working_text)

        return working_text, placeholder_to_original, pii_records

    def unmask_text(self, masked_text: str, pii_mapping: Dict[str, str]) -> str:
        """
        Unmask placeholders for authorized UI display only.
        """
        if not masked_text or not pii_mapping:
            return masked_text

        result = masked_text
        for placeholder, original in pii_mapping.items():
            result = result.replace(placeholder, original)
        return result


pii_masker = PIIMasker()
mask_text = pii_masker.mask_text
unmask_text = pii_masker.unmask_text
