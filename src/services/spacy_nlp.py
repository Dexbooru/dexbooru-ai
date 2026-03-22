"""Load spaCy English pipeline once per process (configured via ApplicationSettings)."""

import spacy
from spacy.language import Language

from utils.config import ApplicationSettings


def load_spacy_english(settings: ApplicationSettings) -> Language:
    """Load the configured English model with the same components disabled as training preprocessing."""
    return spacy.load(settings.spacy_english_model, disable=["ner", "parser"])
