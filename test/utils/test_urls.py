"""Unit tests for utils.urls."""

from utils.urls import is_allowed_net_loc


class TestIsAllowedNetLoc:
    """Tests for is_allowed_net_loc."""

    def test_returns_true_when_netloc_matches(self) -> None:
        assert is_allowed_net_loc("https://cdn.example.com/img.png", "cdn.example.com") is True

    def test_returns_true_when_netloc_matches_case_insensitively_on_url(self) -> None:
        assert is_allowed_net_loc("https://CDN.Example.COM/img.png", "cdn.example.com") is True

    def test_returns_false_when_netloc_does_not_match(self) -> None:
        assert is_allowed_net_loc("https://evil.example.com/img.png", "cdn.example.com") is False

    def test_returns_false_when_expected_netloc_has_different_casing(self) -> None:
        # URL netloc is lowercased; expected_netloc is compared as provided.
        assert is_allowed_net_loc("https://cdn.example.com/img.png", "CDN.example.com") is False

    def test_returns_false_for_url_without_netloc(self) -> None:
        assert is_allowed_net_loc("/relative/path.png", "cdn.example.com") is False

    def test_ignores_path_query_and_fragment(self) -> None:
        assert (
            is_allowed_net_loc(
                "https://cdn.example.com/a/b.png?x=1#frag",
                "cdn.example.com",
            )
            is True
        )

    def test_includes_port_in_netloc_comparison(self) -> None:
        assert is_allowed_net_loc("https://cdn.example.com:8443/img.png", "cdn.example.com:8443") is True
        assert is_allowed_net_loc("https://cdn.example.com:8443/img.png", "cdn.example.com") is False
