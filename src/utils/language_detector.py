"""Language detection and validation utilities."""

import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Automatic language detection for text."""
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ja', 'ko',
        'ru', 'pt', 'it', 'nl', 'pl', 'tr', 'vi', 'th', 'id'
    }
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize language detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.detector = None
        
        # Try to import langdetect
        try:
            from langdetect import detect_langs
            self.detector = detect_langs
            logger.info("Initialized langdetect for language detection")
        except ImportError:
            logger.warning(
                "langdetect not installed, falling back to basic detection. "
                "Install with: pip install langdetect"
            )
    
    def detect(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or not text.strip():
            return None, 0.0
        
        # Use langdetect if available
        if self.detector:
            try:
                results = self.detector(text)
                
                if results:
                    best_result = results[0]
                    lang_code = best_result.lang
                    confidence = best_result.prob
                    
                    # Map to 2-letter codes if needed
                    lang_code = self._normalize_language_code(lang_code)
                    
                    return lang_code, confidence
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        # Fall back to basic script detection
        return self._detect_by_script(text)
    
    def detect_with_validation(
        self,
        text: str,
        expected_lang: Optional[str] = None
    ) -> Tuple[str, float, bool]:
        """
        Detect language and validate against expected language.
        
        Args:
            text: Input text
            expected_lang: Expected language code
            
        Returns:
            Tuple of (detected_lang, confidence, is_valid)
        """
        detected_lang, confidence = self.detect(text)
        
        if detected_lang is None:
            return "unknown", 0.0, False
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return detected_lang, confidence, False
        
        # Validate against expected language if provided
        if expected_lang:
            is_valid = detected_lang == expected_lang
            return detected_lang, confidence, is_valid
        
        return detected_lang, confidence, True
    
    def _detect_by_script(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect language by script (fallback method).
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Count characters by script
        latin_count = 0
        cyrillic_count = 0
        arabic_count = 0
        cjk_count = 0
        hangul_count = 0
        devanagari_count = 0
        
        for char in text:
            code_point = ord(char)
            
            if 0x0041 <= code_point <= 0x024F:  # Latin
                latin_count += 1
            elif 0x0400 <= code_point <= 0x04FF:  # Cyrillic
                cyrillic_count += 1
            elif 0x0600 <= code_point <= 0x06FF:  # Arabic
                arabic_count += 1
            elif (0x4E00 <= code_point <= 0x9FFF or  # CJK Unified
                  0x3040 <= code_point <= 0x309F or  # Hiragana
                  0x30A0 <= code_point <= 0x30FF):   # Katakana
                cjk_count += 1
            elif 0xAC00 <= code_point <= 0xD7AF:  # Hangul
                hangul_count += 1
            elif 0x0900 <= code_point <= 0x097F:  # Devanagari
                devanagari_count += 1
        
        total_chars = (latin_count + cyrillic_count + arabic_count + 
                      cjk_count + hangul_count + devanagari_count)
        
        if total_chars == 0:
            return None, 0.0
        
        # Determine dominant script
        max_count = max(latin_count, cyrillic_count, arabic_count, 
                       cjk_count, hangul_count, devanagari_count)
        
        confidence = max_count / total_chars
        
        # Map script to language (rough approximation)
        if max_count == latin_count:
            return 'en', confidence  # Default to English for Latin
        elif max_count == cyrillic_count:
            return 'ru', confidence
        elif max_count == arabic_count:
            return 'ar', confidence
        elif max_count == cjk_count:
            return 'zh', confidence  # Could be Chinese or Japanese
        elif max_count == hangul_count:
            return 'ko', confidence
        elif max_count == devanagari_count:
            return 'hi', confidence
        
        return None, 0.0
    
    def _normalize_language_code(self, lang_code: str) -> str:
        """
        Normalize language code to 2-letter format.
        
        Args:
            lang_code: Language code (2 or 3 letters)
            
        Returns:
            Normalized 2-letter code
        """
        # Map 3-letter codes to 2-letter codes
        mapping = {
            'eng': 'en',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de',
            'zho': 'zh',
            'ara': 'ar',
            'hin': 'hi',
            'jpn': 'ja',
            'kor': 'ko',
            'rus': 'ru',
            'por': 'pt',
            'ita': 'it',
            'nld': 'nl',
            'pol': 'pl',
            'tur': 'tr',
            'vie': 'vi',
            'tha': 'th',
            'ind': 'id'
        }
        
        return mapping.get(lang_code, lang_code)


class LanguagePairValidator:
    """Validate language pairs against supported combinations."""
    
    # Supported question languages
    QUESTION_LANGUAGES = {'en', 'es', 'fr', 'de', 'zh', 'ar'}
    
    # Supported context languages
    CONTEXT_LANGUAGES = {'en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ja', 'ko'}
    
    def __init__(self):
        """Initialize language pair validator."""
        self.supported_pairs = self._generate_supported_pairs()
    
    def _generate_supported_pairs(self) -> set:
        """
        Generate all supported language pairs.
        
        Returns:
            Set of (question_lang, context_lang) tuples
        """
        pairs = set()
        
        for q_lang in self.QUESTION_LANGUAGES:
            for c_lang in self.CONTEXT_LANGUAGES:
                pairs.add((q_lang, c_lang))
        
        return pairs
    
    def validate_language_code(
        self,
        lang_code: str,
        language_type: str = "question"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a language code.
        
        Args:
            lang_code: Language code to validate
            language_type: "question" or "context"
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if language_type == "question":
            valid_languages = self.QUESTION_LANGUAGES
        elif language_type == "context":
            valid_languages = self.CONTEXT_LANGUAGES
        else:
            return False, f"Invalid language type: {language_type}"
        
        if lang_code not in valid_languages:
            return False, (
                f"Unsupported {language_type} language: {lang_code}. "
                f"Supported languages: {', '.join(sorted(valid_languages))}"
            )
        
        return True, None
    
    def validate_language_pair(
        self,
        question_lang: str,
        context_lang: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a language pair.
        
        Args:
            question_lang: Question language code
            context_lang: Context language code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate individual languages
        q_valid, q_error = self.validate_language_code(question_lang, "question")
        if not q_valid:
            return False, q_error
        
        c_valid, c_error = self.validate_language_code(context_lang, "context")
        if not c_valid:
            return False, c_error
        
        # Check if pair is supported
        pair = (question_lang, context_lang)
        if pair not in self.supported_pairs:
            return False, (
                f"Language pair ({question_lang}, {context_lang}) is not supported. "
                f"Total supported pairs: {len(self.supported_pairs)}"
            )
        
        return True, None
    
    def get_supported_languages(
        self,
        language_type: Optional[str] = None
    ) -> List[str]:
        """
        Get list of supported languages.
        
        Args:
            language_type: "question", "context", or None for all
            
        Returns:
            List of language codes
        """
        if language_type == "question":
            return sorted(self.QUESTION_LANGUAGES)
        elif language_type == "context":
            return sorted(self.CONTEXT_LANGUAGES)
        else:
            return sorted(self.QUESTION_LANGUAGES | self.CONTEXT_LANGUAGES)
    
    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        Get list of all supported language pairs.
        
        Returns:
            List of (question_lang, context_lang) tuples
        """
        return sorted(self.supported_pairs)
    
    def get_pairs_for_language(
        self,
        lang_code: str,
        as_question: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Get all supported pairs for a specific language.
        
        Args:
            lang_code: Language code
            as_question: If True, get pairs where lang is question language
            
        Returns:
            List of language pairs
        """
        pairs = []
        
        if as_question and lang_code in self.QUESTION_LANGUAGES:
            for c_lang in self.CONTEXT_LANGUAGES:
                pairs.append((lang_code, c_lang))
        elif not as_question and lang_code in self.CONTEXT_LANGUAGES:
            for q_lang in self.QUESTION_LANGUAGES:
                pairs.append((q_lang, lang_code))
        
        return sorted(pairs)


# Singleton instances
_language_detector = None
_language_pair_validator = None


def get_language_detector() -> LanguageDetector:
    """Get singleton language detector instance."""
    global _language_detector
    
    if _language_detector is None:
        _language_detector = LanguageDetector()
    
    return _language_detector


def get_language_pair_validator() -> LanguagePairValidator:
    """Get singleton language pair validator instance."""
    global _language_pair_validator
    
    if _language_pair_validator is None:
        _language_pair_validator = LanguagePairValidator()
    
    return _language_pair_validator
