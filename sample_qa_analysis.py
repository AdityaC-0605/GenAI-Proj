#!/usr/bin/env python3
"""
Script to test models with sample QA examples.
Can be used for quick analysis and comparison.
"""

import json
import sys
from pathlib import Path
import requests
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress urllib3 warnings (harmless on macOS)
try:
    from src.utils.warning_suppressor import suppress_urllib3_warnings
    suppress_urllib3_warnings()
except ImportError:
    pass

def load_samples(file_path: str = "sample_qa_examples.json") -> List[Dict]:
    """Load sample QA examples from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['examples']

def test_with_api(question: str, context: str, q_lang: str, c_lang: str, 
                  model: str = "mbert", api_url: str = "http://localhost:8000/predict") -> Dict:
    """
    Test a QA example using the API.
    
    Args:
        question: Question text
        context: Context text
        q_lang: Question language code
        c_lang: Context language code
        model: Model to use ('mbert' or 'mt5')
        api_url: API endpoint URL
        
    Returns:
        API response dictionary
    """
    payload = {
        "question": question,
        "context": context,
        "question_language": q_lang,
        "context_language": c_lang,
        "model_name": model.lower()
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}

def analyze_samples(samples: List[Dict], model: str = "mbert", 
                    api_url: str = "http://localhost:8000/predict"):
    """
    Analyze all samples with a given model.
    
    Args:
        samples: List of sample QA examples
        model: Model to use ('mbert' or 'mt5')
        api_url: API endpoint URL
    """
    print(f"\n{'='*80}")
    print(f"Testing {len(samples)} samples with {model.upper()}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['id']}: {sample['language_pair']}")
        print(f"  Question ({sample['question_language']}): {sample['question']}")
        print(f"  Expected Answer: {sample['answer']}")
        
        # Test with API
        result = test_with_api(
            question=sample['question'],
            context=sample['context'],
            q_lang=sample['question_language'],
            c_lang=sample['context_language'],
            model=model,
            api_url=api_url
        )
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}\n")
            results.append({
                'sample_id': sample['id'],
                'correct': False,
                'error': result['error']
            })
        else:
            predicted = result.get('answer', '')
            confidence = result.get('confidence', 0.0)
            processing_time = result.get('processing_time_ms', 0.0)
            
            # Simple correctness check (case-insensitive, partial match)
            is_correct = sample['answer'].lower() in predicted.lower() or \
                        predicted.lower() in sample['answer'].lower()
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} Predicted: {predicted}")
            print(f"     Confidence: {confidence:.2%}")
            print(f"     Time: {processing_time:.0f}ms")
            print()
            
            results.append({
                'sample_id': sample['id'],
                'language_pair': sample['language_pair'],
                'question_type': sample.get('question_type', 'unknown'),
                'difficulty': sample.get('difficulty', 'unknown'),
                'expected': sample['answer'],
                'predicted': predicted,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'correct': is_correct
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # By language pair
    print("\nBy Language Pair:")
    lang_pair_stats = {}
    for r in results:
        if 'language_pair' in r:
            pair = r['language_pair']
            if pair not in lang_pair_stats:
                lang_pair_stats[pair] = {'total': 0, 'correct': 0}
            lang_pair_stats[pair]['total'] += 1
            if r.get('correct', False):
                lang_pair_stats[pair]['correct'] += 1
    
    for pair, stats in sorted(lang_pair_stats.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {pair}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # By difficulty
    print("\nBy Difficulty:")
    difficulty_stats = {}
    for r in results:
        diff = r.get('difficulty', 'unknown')
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {'total': 0, 'correct': 0}
        difficulty_stats[diff]['total'] += 1
        if r.get('correct', False):
            difficulty_stats[diff]['correct'] += 1
    
    for diff, stats in sorted(difficulty_stats.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {diff}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    return results

def compare_models(samples: List[Dict], api_url: str = "http://localhost:8000/predict"):
    """Compare mBERT and mT5 on the same samples."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Test with mBERT
    mbert_results = analyze_samples(samples, model="mbert", api_url=api_url)
    
    # Test with mT5
    mt5_results = analyze_samples(samples, model="mt5", api_url=api_url)
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    mbert_correct = sum(1 for r in mbert_results if r.get('correct', False))
    mt5_correct = sum(1 for r in mt5_results if r.get('correct', False))
    total = len(mbert_results)
    
    mbert_acc = (mbert_correct / total * 100) if total > 0 else 0
    mt5_acc = (mt5_correct / total * 100) if total > 0 else 0
    
    print(f"mBERT: {mbert_correct}/{total} ({mbert_acc:.1f}%)")
    print(f"mT5:   {mt5_correct}/{total} ({mt5_acc:.1f}%)")
    print(f"Difference: {mt5_acc - mbert_acc:+.1f}%")
    
    # Average confidence
    mbert_conf = sum(r.get('confidence', 0) for r in mbert_results) / total if total > 0 else 0
    mt5_conf = sum(r.get('confidence', 0) for r in mt5_results) / total if total > 0 else 0
    
    print(f"\nAverage Confidence:")
    print(f"mBERT: {mbert_conf:.2%}")
    print(f"mT5:   {mt5_conf:.2%}")
    
    # Average processing time
    mbert_time = sum(r.get('processing_time_ms', 0) for r in mbert_results) / total if total > 0 else 0
    mt5_time = sum(r.get('processing_time_ms', 0) for r in mt5_results) / total if total > 0 else 0
    
    print(f"\nAverage Processing Time:")
    print(f"mBERT: {mbert_time:.0f}ms")
    print(f"mT5:   {mt5_time:.0f}ms")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test models with sample QA examples")
    parser.add_argument('--model', choices=['mbert', 'mt5', 'both'], default='both',
                       help='Model to test (default: both)')
    parser.add_argument('--samples', type=str, default='sample_qa_examples.json',
                       help='Path to samples JSON file')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000/predict',
                       help='API endpoint URL')
    parser.add_argument('--filter', type=str, choices=['cross-lingual', 'same-language', 'all'],
                       default='all', help='Filter samples by type')
    
    args = parser.parse_args()
    
    # Load samples
    samples = load_samples(args.samples)
    
    # Filter samples
    if args.filter == 'cross-lingual':
        samples = [s for s in samples if s['question_language'] != s['context_language']]
    elif args.filter == 'same-language':
        samples = [s for s in samples if s['question_language'] == s['context_language']]
    
    print(f"Loaded {len(samples)} samples (filter: {args.filter})")
    
    # Test models
    if args.model == 'both':
        compare_models(samples, api_url=args.api_url)
    else:
        analyze_samples(samples, model=args.model, api_url=args.api_url)

if __name__ == "__main__":
    main()

