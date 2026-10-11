import ipaddress
from collections.abc import Iterable

from fastapi import Request

UNKNOWN_CLIENT_IP = "unknown"


def _is_trusted_proxy(peer: str, trusted_proxy_ips: Iterable[str]) -> bool:
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for entry in trusted_proxy_ips:
        try:
            if peer_address in ipaddress.ip_network(entry.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def get_client_ip(request: Request, trusted_proxy_ips: Iterable[str]) -> str:
    """Return the client IP, honouring ``X-Forwarded-For`` only when the direct peer is a trusted proxy.

    Trusted proxies may be listed as addresses or CIDR ranges. The left-most forwarded address is used,
    so the trusted proxy must overwrite (not append to) any ``X-Forwarded-For`` sent by the client.
    """
    if request.client is None or not request.client.host:
        return UNKNOWN_CLIENT_IP
    peer = request.client.host

    if not _is_trusted_proxy(peer, trusted_proxy_ips):
        return peer

    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if not forwarded_for:
        return peer

    candidate = forwarded_for.split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer
