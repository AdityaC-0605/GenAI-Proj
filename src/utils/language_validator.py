"""Language validation and detection utilities."""

import re
from typing import Tuple, Optional


def detect_language(text: str) -> str:
    """
    Detect the language of text based on character patterns.
    
    Args:
        text: Input text
        
    Returns:
        Detected language code
    """
    if not text or not text.strip():
        return "en"
    
    # Count characters by script
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
    hindi_chars = len(re.findall(r'[\u0900-\u097f]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    
    total_chars = len(text)
    
    # If more than 20% of characters are from a specific script, detect that language
    threshold = 0.2
    
    if chinese_chars / total_chars > threshold:
        return "zh"
    elif arabic_chars / total_chars > threshold:
        return "ar"
    elif hindi_chars / total_chars > threshold:
        return "hi"
    elif japanese_chars / total_chars > threshold:
        return "ja"
    elif korean_chars / total_chars > threshold:
        return "ko"
    
    # Check for common Spanish/French/German patterns
    spanish_patterns = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
    french_patterns = ['à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü']
    german_patterns = ['ä', 'ö', 'ü', 'ß']
    
    text_lower = text.lower()
    
    spanish_count = sum(1 for p in spanish_patterns if p in text_lower)
    french_count = sum(1 for p in french_patterns if p in text_lower)
    german_count = sum(1 for p in german_patterns if p in text_lower)
    
    if spanish_count > 2:
        return "es"
    elif french_count > 2:
        return "fr"
    elif german_count > 2:
        return "de"
    
    # Default to English
    return "en"


def validate_language_match(
    text: str,
    declared_language: str,
    tolerance: float = 0.3
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate if text matches the declared language.
    
    Args:
        text: Input text
        declared_language: Language code declared by user
        tolerance: Tolerance for mismatch (0.0 to 1.0)
        
    Returns:
        Tuple of (is_valid, detected_language, warning_message)
    """
    detected = detect_language(text)
    
    if detected == declared_language:
        return True, detected, None
    
    # Check if it's a reasonable mismatch (e.g., English text with some Spanish words)
    if declared_language == "en" and detected in ["es", "fr", "de"]:
        # Latin script languages can have some overlap
        return True, detected, f"Note: Text appears to be {detected}, but {declared_language} selected"
    
    # Significant mismatch
    warning = f"⚠️ Language mismatch: Context appears to be in {get_language_name(detected)}, but {get_language_name(declared_language)} was selected. Results may be inaccurate."
    
    return False, detected, warning


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    language_names = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ar": "Arabic",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean"
    }
    return language_names.get(code, code.upper())


def should_reduce_confidence(
    question_lang: str,
    context_lang: str,
    detected_context_lang: str
) -> Tuple[bool, float]:
    """
    Determine if confidence should be reduced due to language mismatch.
    
    Args:
        question_lang: Declared question language
        context_lang: Declared context language
        detected_context_lang: Detected context language
        
    Returns:
        Tuple of (should_reduce, penalty_factor)
    """
    # If declared and detected don't match, reduce confidence
    if context_lang != detected_context_lang:
        # Severe mismatch (e.g., Chinese context labeled as English)
        if detected_context_lang in ["zh", "ar", "hi", "ja", "ko"] and context_lang == "en":
            return True, 0.6  # Reduce confidence by 40%
        # Moderate mismatch (e.g., Spanish labeled as English)
        else:
            return True, 0.8  # Reduce confidence by 20%
    
    return False, 1.0  # No reduction
