"""
Unit tests for Server-Side PII Masker (Requirement 2 & 3).
"""

import pytest
from ai.app.services.pii_masker import PIIMasker, mask_text, unmask_text


@pytest.fixture
def masker():
    return PIIMasker()


def test_mask_email(masker):
    raw = "Please contact the client at john.doe@example.com regarding portfolio updates."
    masked, mapping, records = masker.mask_text(raw)
    assert "[EMAIL_1]" in masked
    assert "john.doe@example.com" not in masked
    assert mapping["[EMAIL_1]"] == "john.doe@example.com"


def test_mask_phone_numbers(masker):
    raw = "Call our advisor at (555) 987-6543 or mobile +1-555-123-4567."
    masked, mapping, records = masker.mask_text(raw)
    assert "[PHONE_1]" in masked
    assert "(555) 987-6543" not in masked
    assert "+1-555-123-4567" not in masked


def test_mask_ssn(masker):
    raw = "Client tax identifier is 123-45-6789 on the custody document."
    masked, mapping, records = masker.mask_text(raw)
    assert "[SSN_1]" in masked
    assert "123-45-6789" not in masked
    assert mapping["[SSN_1]"] == "123-45-6789"


def test_mask_account_numbers(masker):
    raw = "Transfer funds into Account #987654321 at the clearing firm."
    masked, mapping, records = masker.mask_text(raw)
    assert "[ACCOUNT_1]" in masked
    assert "987654321" not in masked


def test_mask_address(masker):
    raw = "Send the quarterly statements to 450 Lexington Avenue, Suite 1200, New York, NY 10017."
    masked, mapping, records = masker.mask_text(raw)
    assert "[ADDRESS_1]" in masked
    assert "450 Lexington Avenue" not in masked


def test_mask_names_with_context(masker):
    raw = "Client: Alice Walker reviewed the prospectus. Dear Robert Taylor, welcome."
    masked, mapping, records = masker.mask_text(raw)
    assert "[CLIENT_1]" in masked
    assert "[CLIENT_2]" in masked
    assert "Alice Walker" not in masked
    assert "Robert Taylor" not in masked


def test_stable_placeholder_consistency(masker):
    raw = "Client: Alice Walker submitted the document. Alice Walker confirmed the transfer for Account #12345678. Later Alice Walker called about Account #12345678."
    masked, mapping, records = masker.mask_text(raw)
    # Alice Walker should consistently map to [CLIENT_1]
    assert masked.count("[CLIENT_1]") >= 2
    # Account should consistently map to [ACCOUNT_1]
    assert masked.count("[ACCOUNT_1]") == 2
    assert "Alice Walker" not in masked
    assert "12345678" not in masked


def test_multiple_clients(masker):
    raw = "Client: Alice Walker and Investor: Bob Johnson both opened accounts."
    masked, mapping, records = masker.mask_text(raw)
    assert "[CLIENT_1]" in masked
    assert "[CLIENT_2]" in masked
    assert mapping["[CLIENT_1]"] == "Alice Walker"
    assert mapping["[CLIENT_2]"] == "Bob Johnson"


def test_tied_dollar_amounts(masker):
    raw = "Client: John Smith is managing a portfolio of $2,500,000 in equity assets."
    masked, mapping, records = masker.mask_text(raw)
    assert "[AMOUNT_1]" in masked or "[CLIENT_1]" in masked
    assert "John Smith" not in masked


def test_unmask_text(masker):
    raw = "Client: Jane Doe with Account #99887766 requested withdrawal to jane.doe@example.com."
    masked, mapping, _ = masker.mask_text(raw)
    assert "[CLIENT_1]" in masked
    assert "[ACCOUNT_1]" in masked
    assert "[EMAIL_1]" in masked

    unmasked = masker.unmask_text(masked, mapping)
    assert "Jane Doe" in unmasked
    assert "99887766" in unmasked
    assert "jane.doe@example.com" in unmasked


def test_no_pii_text(masker):
    raw = "All investing involves risk of loss. Past performance is no guarantee of future results."
    masked, mapping, records = masker.mask_text(raw)
    assert masked == raw
    assert len(mapping) == 0
    assert len(records) == 0


def test_false_positive_exclusions(masker):
    raw = "The Compliance Officer and Financial Advisor reviewed the Mutual Fund performance in the United States."
    masked, mapping, records = masker.mask_text(raw)
    # None of these standard financial terms should be masked
    assert "Compliance Officer" in masked
    assert "Financial Advisor" in masked
    assert "Mutual Fund" in masked
    assert "United States" in masked
    assert len(mapping) == 0


def test_step3_synthetic_document_masking(masker):
    test_doc = """Client Name: John Smith
Email: john.smith@example.com
Phone: 9876543210
Account Number: 123456789012
Address: 12 MG Road, Hyderabad
Investment Amount: $50,000

John Smith contacted us again.
john.smith@example.com
Account Number: 123456789012"""

    masked, mapping, records = masker.mask_text(test_doc, "doc_test_step3")

    # 1. Assert all original PII is absent
    assert "John Smith" not in masked
    assert "john.smith@example.com" not in masked
    assert "9876543210" not in masked
    assert "123456789012" not in masked
    assert "12 MG Road, Hyderabad" not in masked
    assert "$50,000" not in masked

    # 2. Assert placeholders are correctly generated
    assert "[CLIENT_1]" in masked
    assert "[EMAIL_1]" in masked
    assert "[PHONE_1]" in masked
    assert "[ACCOUNT_1]" in masked
    assert "[ADDRESS_1]" in masked
    assert "[AMOUNT_1]" in masked

    # 3. Assert repeated occurrences use the identical placeholder
    assert masked.count("[CLIENT_1]") == 2
    assert masked.count("[EMAIL_1]") == 2
    assert masked.count("[ACCOUNT_1]") == 2

    # 4. Verify server-side mapping integrity
    assert mapping["[CLIENT_1]"] == "John Smith"
    assert mapping["[EMAIL_1]"] == "john.smith@example.com"
    assert mapping["[PHONE_1]"] == "9876543210"
    assert mapping["[ACCOUNT_1]"] == "123456789012"
    assert mapping["[ADDRESS_1]"] == "12 MG Road, Hyderabad"
    assert mapping["[AMOUNT_1]"] == "$50,000"

