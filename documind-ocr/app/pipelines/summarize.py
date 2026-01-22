from typing import List
from collections import Counter
import re

def summarize_document(text: str) -> List[str]:
    """
    Generate a simple extractive summary (3-5 bullet points).
    Since we can't use LLMs, we'll use frequency-based or lead-based sentences.
    """
    # 1. Split into sentences (naive)
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return ["No text content sufficiently long to summarize."]
    
    # 2. Simple heuristic: Pick lines with monetary values or dates (likely important)
    # OR: Pick first and last few lines
    
    important_sentences = []
    
    # Priority 1: Sentences with numbers (prices)
    for s in sentences:
        if re.search(r'\d', s):
            important_sentences.append(s)
            
    # Priority 2: Use top sentences if not enough
    if len(important_sentences) < 3:
        remaining = [s for s in sentences if s not in important_sentences]
        important_sentences.extend(remaining[:3])
        
    # Limit to 3 bullets
    summary = important_sentences[:3]
    return summary
