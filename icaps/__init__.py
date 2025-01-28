import json
from pathlib import Path

__version__ = "0.0.1"

def load_credentials(base: Path = None) -> dict:
    """
    Loads the OpenReview credentials
    :return:
    """
    if base is None:
        base = Path(".")

    with open(base / 'credentials.json') as credentials_file:
        credentials = json.load(credentials_file)
        return credentials