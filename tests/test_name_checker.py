import pytest

from utils.name_checker import check_plausibility


def _pct(x) -> float:
    """Accepts numeric-like values and returns float percent."""
    return float(x)


@pytest.mark.parametrize(
    "first, last, claimed_country, expected_ratio_min, expected_ratio_max, expected_label_any",
    [
        # Typical German name in Germany -> plausibility should be >= ~neutral/typical
        ("Anna", "Müller", "Germany", 0.7, 1000.0, ["Neutral", "Typical", "Very typical"]),

        # Typical Polish name in Poland
        ("Zofia", "Kowalska", "Poland", 0.7, 1000.0, ["Neutral", "Typical", "Very typical"]),

        # German name claimed in Poland -> plausibility often lower than Germany case
        # (we check it's not "Very typical" and plausibility ratio not huge)
        ("Anna", "Müller", "Poland", 0.0, 3.0, ["Very unusual", "Unusual", "Neutral", "Typical"]),

        # Non-existing name everywhere -> very low plausibility, low probability share
        ("Xzqwerty", "Blablablinsky", "Germany", 0.0, 1.5, ["Very unusual", "Unusual", "Neutral"]),

        # Rare but exists -> typically unusual/neutral
        ("Einar", "Björnsson", "Germany", 0.0, 3.0, ["Very unusual", "Unusual", "Neutral", "Typical"]),

        # Common-ish global name; depends on dataset coverage for UK
        ("Mohammed", "Ali", "United Kingdom", 0.0, 1000.0, ["Unusual", "Neutral", "Typical", "Very typical"]),

        # Edge case: empty first name (should not crash)
        ("", "Schmidt", "Germany", 0.0, 1000.0, ["Very unusual", "Unusual", "Neutral", "Typical", "Very typical"]),
    ],
)
def test_plausibility_scenarios(
    first,
    last,
    claimed_country,
    expected_ratio_min,
    expected_ratio_max,
    expected_label_any,
):
    result = check_plausibility(first, last, claimed_country)

    assert isinstance(result, dict)

    # New keys
    assert "plausibility_ratio" in result
    assert "plausibility_label" in result
    assert "posterior_share_claimed_pct" in result
    assert "claimed_rank" in result
    assert "top_country" in result
    assert "ranked_countries" in result
    assert "country" in result

    assert result["country"] == claimed_country

    ratio = float(result["plausibility_ratio"])
    assert expected_ratio_min <= ratio <= expected_ratio_max, (
        f"plausibility_ratio {ratio} not in expected range "
        f"[{expected_ratio_min}, {expected_ratio_max}]"
    )

    # Label should be one of the known labels
    assert result["plausibility_label"] in {
        "Very unusual",
        "Unusual",
        "Neutral",
        "Typical",
        "Very typical",
    }

    # Scenario expects label to be within some acceptable set
    assert result["plausibility_label"] in set(expected_label_any), (
        f"Label '{result['plausibility_label']}' not in allowed set {expected_label_any}"
    )

    # Posterior share percent sanity
    p = _pct(result["posterior_share_claimed_pct"])
    assert 0.0 <= p <= 100.0

    # ranked_countries shape / contents
    ranked = result["ranked_countries"]
    assert isinstance(ranked, list)
    assert 0 < len(ranked) <= 8
    for row in ranked:
        assert isinstance(row, dict)
        assert {"rank", "country", "posterior_share_pct", "first_count", "last_count", "is_claimed"} <= set(row.keys())
        assert isinstance(row["rank"], int)
        assert 1 <= row["rank"] <= 8
        assert isinstance(row["country"], str)
        assert 0.0 <= float(row["posterior_share_pct"]) <= 100.0
        assert isinstance(row["first_count"], int)
        assert isinstance(row["last_count"], int)
        assert isinstance(row["is_claimed"], bool)

    # Claimed country should appear in full per_country ranking, but might not appear in top 8
    # So only check that `claimed_rank` is either "unknown" or a positive int.
    cr = result["claimed_rank"]
    assert cr == "unknown" or (isinstance(cr, int) and cr >= 1)

    # top_country sanity
    assert isinstance(result["top_country"], str)
    assert len(result["top_country"]) > 0


def test_case_insensitivity():
    result_upper = check_plausibility("ANNA", "MÜLLER", "Germany")
    result_lower = check_plausibility("anna", "müller", "Germany")

    assert float(result_upper["plausibility_ratio"]) == float(result_lower["plausibility_ratio"])
    assert result_upper["plausibility_label"] == result_lower["plausibility_label"]
    assert float(result_upper["posterior_share_claimed_pct"]) == float(result_lower["posterior_share_claimed_pct"])