"""Error analysis module for Cross-Lingual QA models."""

import logging
from typing import List, Dict, Any, Tuple
from enum import Enum
from collections import defaultdict, Counter

from src.data_models import QAExample, QAPrediction

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for QA predictions."""
    NO_ANSWER = "no_answer"
    PARTIAL_ANSWER = "partial_answer"
    INCORRECT_ANSWER = "incorrect_answer"
    WRONG_LANGUAGE = "wrong_language"
    HALLUCINATION = "hallucination"


class ErrorCategorizer:
    """Categorize prediction errors."""
    
    def __init__(self, min_overlap_threshold: float = 0.3):
        """
        Initialize error categorizer.
        
        Args:
            min_overlap_threshold: Minimum token overlap for partial answer
        """
        self.min_overlap_threshold = min_overlap_threshold
    
    def categorize_error(
        self,
        prediction: QAPrediction,
        example: QAExample
    ) -> ErrorCategory:
        """
        Categorize a prediction error.
        
        Args:
            prediction: Model prediction
            example: Ground truth example
            
        Returns:
            ErrorCategory enum value
        """
        pred_text = prediction.answer_text.strip()
        
        # Check for no answer
        if not pred_text or pred_text.lower() in ["", "no answer", "none"]:
            return ErrorCategory.NO_ANSWER
        
        # Check for wrong language (basic heuristic)
        if self._is_wrong_language(pred_text, example):
            return ErrorCategory.WRONG_LANGUAGE
        
        # Check for hallucination (answer not in context)
        if not self._answer_in_context(pred_text, example.context):
            return ErrorCategory.HALLUCINATION
        
        # Check for partial vs incorrect answer
        ground_truth_texts = [ans.text for ans in example.answers]
        max_overlap = max(
            self._calculate_token_overlap(pred_text, gt)
            for gt in ground_truth_texts
        )
        
        if max_overlap >= self.min_overlap_threshold:
            return ErrorCategory.PARTIAL_ANSWER
        else:
            return ErrorCategory.INCORRECT_ANSWER
    
    def _is_wrong_language(self, prediction: str, example: QAExample) -> bool:
        """
        Check if prediction is in wrong language (basic heuristic).
        
        Args:
            prediction: Predicted answer text
            example: Ground truth example
            
        Returns:
            True if prediction appears to be in wrong language
        """
        # Simple heuristic: check if prediction uses different script
        pred_script = self._detect_script(prediction)
        context_script = self._detect_script(example.context)
        
        # If scripts are very different, likely wrong language
        return pred_script != context_script and pred_script != "mixed"
    
    def _detect_script(self, text: str) -> str:
        """
        Detect script type of text.
        
        Args:
            text: Input text
            
        Returns:
            Script type: latin, cyrillic, arabic, cjk, or mixed
        """
        if not text:
            return "unknown"
        
        latin_count = sum(1 for c in text if ord(c) < 0x0250)
        cyrillic_count = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
        arabic_count = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
        cjk_count = sum(
            1 for c in text
            if 0x4E00 <= ord(c) <= 0x9FFF or  # CJK Unified
               0x3040 <= ord(c) <= 0x309F or  # Hiragana
               0x30A0 <= ord(c) <= 0x30FF or  # Katakana
               0xAC00 <= ord(c) <= 0xD7AF     # Hangul
        )
        
        total_chars = len([c for c in text if c.isalpha()])
        if total_chars == 0:
            return "unknown"
        
        # Determine dominant script
        scripts = {
            "latin": latin_count / total_chars,
            "cyrillic": cyrillic_count / total_chars,
            "arabic": arabic_count / total_chars,
            "cjk": cjk_count / total_chars
        }
        
        max_script = max(scripts.items(), key=lambda x: x[1])
        
        if max_script[1] < 0.5:
            return "mixed"
        
        return max_script[0]
    
    def _answer_in_context(self, answer: str, context: str) -> bool:
        """
        Check if answer appears in context.
        
        Args:
            answer: Answer text
            context: Context text
            
        Returns:
            True if answer is found in context
        """
        # Normalize for comparison
        answer_norm = answer.lower().strip()
        context_norm = context.lower()
        
        # Check for exact match
        if answer_norm in context_norm:
            return True
        
        # Check for token-level match (for generative models)
        answer_tokens = set(answer_norm.split())
        context_tokens = set(context_norm.split())
        
        # If most answer tokens are in context, consider it grounded
        if len(answer_tokens) > 0:
            overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
            return overlap > 0.7
        
        return False
    
    def _calculate_token_overlap(self, pred: str, ground_truth: str) -> float:
        """
        Calculate token overlap between prediction and ground truth.
        
        Args:
            pred: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Overlap ratio (0-1)
        """
        pred_tokens = set(pred.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        intersection = pred_tokens & gt_tokens
        union = pred_tokens | gt_tokens
        
        return len(intersection) / len(union) if union else 0.0


class ErrorCorrelationAnalyzer:
    """Analyze error correlations with question types and linguistic distance."""
    
    def __init__(self):
        """Initialize error correlation analyzer."""
        self.question_type_patterns = {
            'what': ['what', 'which'],
            'when': ['when'],
            'where': ['where'],
            'who': ['who', 'whom', 'whose'],
            'why': ['why'],
            'how': ['how']
        }
    
    def analyze_by_question_type(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
        error_categories: List[ErrorCategory]
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze error correlation with question types.
        
        Args:
            examples: List of QA examples
            predictions: List of predictions
            error_categories: List of error categories
            
        Returns:
            Dictionary mapping question types to error counts
        """
        question_type_errors = defaultdict(lambda: defaultdict(int))
        
        for example, prediction, error_cat in zip(examples, predictions, error_categories):
            q_type = self._classify_question_type(example.question)
            question_type_errors[q_type][error_cat.value] += 1
        
        return dict(question_type_errors)
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classify question type based on question word.
        
        Args:
            question: Question text
            
        Returns:
            Question type (what, when, where, who, why, how, other)
        """
        question_lower = question.lower()
        
        for q_type, patterns in self.question_type_patterns.items():
            for pattern in patterns:
                if question_lower.startswith(pattern):
                    return q_type
        
        return 'other'
    
    def generate_confusion_matrix(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
        is_correct: List[bool]
    ) -> Dict[str, Dict[str, int]]:
        """
        Generate confusion matrix by question type.
        
        Args:
            examples: List of QA examples
            predictions: List of predictions
            is_correct: List of correctness flags
            
        Returns:
            Confusion matrix as nested dictionary
        """
        confusion_matrix = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        
        for example, pred, correct in zip(examples, predictions, is_correct):
            q_type = self._classify_question_type(example.question)
            
            if correct:
                confusion_matrix[q_type]['correct'] += 1
            else:
                confusion_matrix[q_type]['incorrect'] += 1
        
        return dict(confusion_matrix)
    
    def analyze_by_linguistic_distance(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
        scores: List[float]
    ) -> Dict[str, List[Tuple[Tuple[str, str], float]]]:
        """
        Analyze error correlation with linguistic distance.
        
        Args:
            examples: List of QA examples
            predictions: List of predictions
            scores: List of F1 scores
            
        Returns:
            Dictionary mapping distance categories to (lang_pair, score) tuples
        """
        # Simple linguistic distance categorization
        distance_categories = {
            'same_language': [],
            'similar_family': [],
            'different_family': []
        }
        
        for example, pred, score in zip(examples, predictions, scores):
            q_lang = example.question_language
            c_lang = example.context_language
            lang_pair = (q_lang, c_lang)
            
            if q_lang == c_lang:
                distance_categories['same_language'].append((lang_pair, score))
            elif self._are_similar_languages(q_lang, c_lang):
                distance_categories['similar_family'].append((lang_pair, score))
            else:
                distance_categories['different_family'].append((lang_pair, score))
        
        return distance_categories
    
    def _are_similar_languages(self, lang1: str, lang2: str) -> bool:
        """
        Check if two languages are from similar families.
        
        Args:
            lang1: First language code
            lang2: Second language code
            
        Returns:
            True if languages are from similar families
        """
        # Language family groupings
        romance = {'es', 'fr', 'it', 'pt', 'ro'}
        germanic = {'en', 'de', 'nl', 'sv', 'da'}
        slavic = {'ru', 'pl', 'cs', 'uk'}
        cjk = {'zh', 'ja', 'ko'}
        
        families = [romance, germanic, slavic, cjk]
        
        for family in families:
            if lang1 in family and lang2 in family:
                return True
        
        return False


class ErrorExampleExtractor:
    """Extract representative error examples."""
    
    def __init__(self, max_examples_per_category: int = 5):
        """
        Initialize error example extractor.
        
        Args:
            max_examples_per_category: Maximum examples to extract per category
        """
        self.max_examples_per_category = max_examples_per_category
    
    def extract_examples(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
        error_categories: List[ErrorCategory]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract representative error examples for each category.
        
        Args:
            examples: List of QA examples
            predictions: List of predictions
            error_categories: List of error categories
            
        Returns:
            Dictionary mapping error categories to example lists
        """
        error_examples = defaultdict(list)
        
        for example, prediction, error_cat in zip(examples, predictions, error_categories):
            if len(error_examples[error_cat.value]) < self.max_examples_per_category:
                error_examples[error_cat.value].append({
                    'question': example.question,
                    'context': example.context[:200] + '...',  # Truncate for readability
                    'ground_truth': [ans.text for ans in example.answers],
                    'prediction': prediction.answer_text,
                    'confidence': prediction.confidence,
                    'question_language': example.question_language,
                    'context_language': example.context_language
                })
        
        return dict(error_examples)
    
    def save_error_examples(
        self,
        error_examples: Dict[str, List[Dict[str, Any]]],
        output_path: str
    ):
        """
        Save error examples to file.
        
        Args:
            error_examples: Dictionary of error examples
            output_path: Path to save examples
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error examples saved to {output_path}")


class ErrorAnalyzer:
    """Main error analysis coordinator."""
    
    def __init__(self):
        """Initialize error analyzer."""
        self.categorizer = ErrorCategorizer()
        self.correlation_analyzer = ErrorCorrelationAnalyzer()
        self.example_extractor = ErrorExampleExtractor()
    
    def analyze(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.
        
        Args:
            examples: List of QA examples
            predictions: List of predictions
            scores: List of F1 scores
            
        Returns:
            Dictionary with complete error analysis
        """
        logger.info(f"Starting error analysis on {len(examples)} examples")
        
        # Categorize errors
        error_categories = []
        is_correct = []
        
        for example, prediction, score in zip(examples, predictions, scores):
            if score < 1.0:  # Not perfect match
                error_cat = self.categorizer.categorize_error(prediction, example)
                error_categories.append(error_cat)
                is_correct.append(False)
            else:
                error_categories.append(None)
                is_correct.append(True)
        
        # Count error distribution
        error_counts = Counter(
            cat.value for cat in error_categories if cat is not None
        )
        
        # Analyze by question type
        question_type_errors = self.correlation_analyzer.analyze_by_question_type(
            examples, predictions, [cat for cat in error_categories if cat is not None]
        )
        
        # Generate confusion matrix
        confusion_matrix = self.correlation_analyzer.generate_confusion_matrix(
            examples, predictions, is_correct
        )
        
        # Analyze by linguistic distance
        linguistic_distance_analysis = self.correlation_analyzer.analyze_by_linguistic_distance(
            examples, predictions, scores
        )
        
        # Extract error examples
        error_examples = self.example_extractor.extract_examples(
            [ex for ex, cat in zip(examples, error_categories) if cat is not None],
            [pred for pred, cat in zip(predictions, error_categories) if cat is not None],
            [cat for cat in error_categories if cat is not None]
        )
        
        analysis_results = {
            'error_distribution': dict(error_counts),
            'question_type_errors': question_type_errors,
            'confusion_matrix': confusion_matrix,
            'linguistic_distance_analysis': {
                'same_language': {
                    'count': len(linguistic_distance_analysis['same_language']),
                    'avg_score': sum(s for _, s in linguistic_distance_analysis['same_language']) / 
                                len(linguistic_distance_analysis['same_language'])
                                if linguistic_distance_analysis['same_language'] else 0.0
                },
                'similar_family': {
                    'count': len(linguistic_distance_analysis['similar_family']),
                    'avg_score': sum(s for _, s in linguistic_distance_analysis['similar_family']) / 
                                len(linguistic_distance_analysis['similar_family'])
                                if linguistic_distance_analysis['similar_family'] else 0.0
                },
                'different_family': {
                    'count': len(linguistic_distance_analysis['different_family']),
                    'avg_score': sum(s for _, s in linguistic_distance_analysis['different_family']) / 
                                len(linguistic_distance_analysis['different_family'])
                                if linguistic_distance_analysis['different_family'] else 0.0
                }
            },
            'error_examples': error_examples
        }
        
        logger.info("Error analysis complete")
        logger.info(f"Error distribution: {error_counts}")
        
        return analysis_results
