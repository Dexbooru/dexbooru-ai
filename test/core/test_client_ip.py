"""Unit tests for core.client_ip.get_client_ip."""

import pytest
from starlette.requests import Request

from core.client_ip import UNKNOWN_CLIENT_IP, get_client_ip


def _make_request(peer: str | None, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (peer, 12345) if peer is not None else None,
    }
    return Request(scope)


PROXY = "10.0.0.2"


class TestGetClientIp:
    def test_trusted_proxy_uses_forwarded_for(self) -> None:
        request = _make_request(PROXY, "203.0.113.7")
        assert get_client_ip(request, [PROXY]) == "203.0.113.7"

    def test_untrusted_peer_ignores_forwarded_for(self) -> None:
        request = _make_request("198.51.100.9", "203.0.113.7")
        assert get_client_ip(request, [PROXY]) == "198.51.100.9"

    def test_no_trusted_proxies_ignores_forwarded_for(self) -> None:
        request = _make_request(PROXY, "203.0.113.7")
        assert get_client_ip(request, []) == PROXY

    def test_multiple_entries_uses_left_most(self) -> None:
        request = _make_request(PROXY, " 203.0.113.7 , 198.51.100.1, 10.0.0.1")
        assert get_client_ip(request, [PROXY]) == "203.0.113.7"

    def test_missing_header_uses_peer(self) -> None:
        request = _make_request(PROXY)
        assert get_client_ip(request, [PROXY]) == PROXY

    @pytest.mark.parametrize("value", ["", "   ", " , "])
    def test_empty_header_uses_peer(self, value: str) -> None:
        request = _make_request(PROXY, value)
        assert get_client_ip(request, [PROXY]) == PROXY

    @pytest.mark.parametrize("value", ["not-an-ip", "203.0.113.7:8080", "999.1.1.1", "<script>"])
    def test_invalid_header_value_uses_peer(self, value: str) -> None:
        request = _make_request(PROXY, value)
        assert get_client_ip(request, [PROXY]) == PROXY

    def test_ipv6_forwarded_for_is_normalized(self) -> None:
        request = _make_request(PROXY, "2001:DB8::0001")
        assert get_client_ip(request, [PROXY]) == "2001:db8::1"

    def test_trusted_proxy_cidr(self) -> None:
        request = _make_request("172.18.0.5", "203.0.113.7")
        assert get_client_ip(request, ["172.18.0.0/16"]) == "203.0.113.7"

    def test_invalid_trusted_entries_are_ignored(self) -> None:
        request = _make_request(PROXY, "203.0.113.7")
        assert get_client_ip(request, ["garbage", PROXY]) == "203.0.113.7"

    def test_non_ip_peer_is_never_trusted(self) -> None:
        request = _make_request("testclient", "203.0.113.7")
        assert get_client_ip(request, ["testclient"]) == "testclient"

    def test_missing_client(self) -> None:
        request = _make_request(None, "203.0.113.7")
        assert get_client_ip(request, [PROXY]) == UNKNOWN_CLIENT_IP
