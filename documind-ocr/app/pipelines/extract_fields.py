import re
from typing import Dict, Any, List
import dateparser
import spacy
from datetime import datetime

# Load Spacy for NER (optional usage)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

def extract_date(text: str, tokens: List[Dict]) -> Dict[str, Any]:
    """Extract Invoice Date."""
    # 1. Look for specific keywords and take the date near it
    date_keywords = ['invoice date', 'date:', 'dated:', 'bill date']
    
    lines = text.split('\n') # This assumes text is newline separated, but OCR might be just space. 
    # Use tokens for better proximity search if needed. For now simple Regex.
    
    # Regex for common formats
    date_regex = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{4}[/-]\d{1,2}[/-]\d{1,2})|([A-Za-z]{3}\s\d{1,2},\s\d{4})'
    
    matches = []
    
    # Heuristic: Check near keywords
    # This is tricky without spatial graph, so we scan full text for candidates.
    candidates = re.finditer(date_regex, text)
    
    best_date = None
    max_score = 0
    
    for match in candidates:
        d_str = match.group(0)
        try:
            dt = dateparser.parse(d_str)
            if dt:
                score = 0.5
                # Proximity boost
                start_idx = match.start()
                context = text[max(0, start_idx-20):start_idx].lower()
                if any(k in context for k in date_keywords):
                    score += 0.4
                
                if score > max_score:
                    max_score = score
                    best_date = dt.strftime("%Y-%m-%d")
        except:
            continue
            
    if best_date:
        return {"value": best_date, "confidence": max_score}
    
    return {"value": None, "confidence": 0.0}

def extract_amount(text: str, tokens: List[Dict]) -> Dict[str, Any]:
    """Extract Total Amount."""
    # Keywords
    total_keywords = ['total', 'grand total', 'amount due', 'balance due']
    
    # Regex for currency: $1,234.56 or 1234.56
    # Avoid picking dates or phone numbers
    amount_regex = r'[\$€£₹]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    
    matches = re.finditer(amount_regex, text)
    
    candidates = []
    for match in matches:
        val_str = match.group(1).replace(',', '')
        try:
            val = float(val_str)
            # Filter unlikely amounts
            if val < 0.01: continue
            
            start_idx = match.start()
            # Look backwards for "Total"
            context = text[max(0, start_idx-40):start_idx].lower()
            
            score = 0.5
            if any(k in context for k in total_keywords):
                score = 0.9 # High confidence if near Total keyword
            elif 'subtotal' in context:
                score = 0.3 # Less likely to be grand total
                
            candidates.append((val, score))
        except:
            continue
    
    if not candidates:
        return {"value": None, "currency": None, "confidence": 0.0}
        
    # Sort by score desc, then value desc (often total is max value)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    best_val, best_score = candidates[0]
    
    return {"value": best_val, "currency": "USD", "confidence": best_score} # Default currency USD for now

def extract_vendor(text: str, tokens: List[Dict]) -> Dict[str, Any]:
    """Extract Vendor Name using Spacy and Layout."""
    # Heuristic: Vendor is often at the top, largest text.
    # tokens are sorted by Y
    
    # 1. Check top 5 lines for ORGS using Spacy
    if nlp:
        doc = nlp(text[:500]) # First 500 chars
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return {"value": ent.text, "confidence": 0.8}
                
    # 2. Fallback: First non-trivial line
    lines = [L.strip() for L in text.split('\n') if len(L.strip()) > 3]
    if lines:
        return {"value": lines[0], "confidence": 0.4}
        
    return {"value": None, "confidence": 0.0}

def extract_fields(text: str, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main extraction routine."""
    
    vendor = extract_vendor(text, tokens)
    date = extract_date(text, tokens)
    amount = extract_amount(text, tokens)
    
    return {
        "vendor_name": vendor,
        "invoice_date": date,
        "total_amount": amount
    }
