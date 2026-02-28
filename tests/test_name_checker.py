import pytest
from utils.name_checker import check_plausibility


@pytest.mark.parametrize(
    "first, last, claimed_country, expected_min_score, expected_max_score, expected_keywords",
    [
        # Very typical German name in Germany -> high score
        ("Anna", "Müller", "Germany", 80, 100, ["Common", "typical"]),

        # Very typical Polish name in Poland -> high score
        ("Zofia", "Kowalska", "Poland", 80, 100, ["Common", "typical"]),

        # German name claimed in Poland -> low score + suspicion hint
        ("Anna", "Müller", "Poland", 5, 45, ["Rare", "uncommon", "suspicious"]),

        # Non-existing name everywhere -> very low score
        ("Xzqwerty", "Blablablinsky", "Germany", 0, 15, ["nowhere"]),

        # Rare name but exists a bit -> medium-low
        ("Einar", "Björnsson", "Germany", 10, 50, ["Rare", "unusual"]),

        # Very common name in claimed country
        ("Mohammed", "Ali", "United Kingdom", 40, 85, ["Common"]),

        # Edge case: empty inputs (should be handled gracefully)
        ("", "Schmidt", "Germany", 0, 100, ["Rare", "zero"]),
    ]
)
def test_plausibility_scenarios(
    first, last, claimed_country,
    expected_min_score, expected_max_score, expected_keywords
):
    result = check_plausibility(first, last, claimed_country)

    assert isinstance(result, dict)
    assert "score" in result
    assert "rarity" in result
    assert "country" in result
    assert "message" in result

    assert result["country"] == claimed_country

    # Score range check
    assert expected_min_score <= result["score"] <= expected_max_score, \
        f"Score {result['score']} not in expected range [{expected_min_score}, {expected_max_score}]"

    # Keyword presence in message (case insensitive)
    msg_lower = result["message"].lower()
    for kw in expected_keywords:
        assert kw.lower() in msg_lower, f"Expected keyword '{kw}' not found in: {result['message']}"

    # Basic sanity
    assert 0 <= result["score"] <= 100
    assert len(result["message"]) > 10


def test_non_european_claim_fallback():
    # If someone sends a non-existing country name → should still run without crash
    result = check_plausibility("Anna", "Novak", "Atlantis")
    assert result["score"] >= 0
    assert "country" in result
    assert isinstance(result["score"], int)


def test_case_insensitivity():
    result_upper = check_plausibility("ANNA", "MÜLLER", "Germany")
    result_lower = check_plausibility("anna", "müller", "Germany")

    assert result_upper["score"] == result_lower["score"]
    assert result_upper["rarity"] == result_lower["rarity"]