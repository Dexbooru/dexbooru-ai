import urllib.parse


def is_allowed_net_loc(raw_url: str, expected_netloc: str) -> bool:
    parsed_url = urllib.parse.urlsplit(raw_url)
    if not parsed_url:
        return False

    parsed_url_netloc = parsed_url.netloc.lower().strip()
    return parsed_url_netloc == expected_netloc
