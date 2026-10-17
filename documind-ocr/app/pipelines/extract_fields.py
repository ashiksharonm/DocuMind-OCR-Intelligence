import re
from typing import Dict, Any, List, Optional
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
    best_raw = None
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
                    best_raw = d_str
        except:
            continue
            
    if best_date:
        return {"value": best_date, "confidence": max_score, "raw": best_raw}
    
    return {"value": None, "confidence": 0.0}

def _labelled(text: str, label: str) -> Optional[float]:
    m = re.search("(?:" + label + r")[^0-9\n]{0,12}(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})", text, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def _sum_check(text: str, total: float) -> Optional[bool]:
    """True/False when a labelled subtotal and tax exist and do/don't add up to ``total``."""
    sub, tax = _labelled(text, r"sub\s?total"), _labelled(text, r"\btax|\bgst|\bvat")
    if sub is None or tax is None:
        return None
    return abs(sub + tax - total) <= 0.011


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
                
            candidates.append((val, score, match.group(1)))
        except:
            continue
    
    if not candidates:
        return {"value": None, "currency": None, "confidence": 0.0}
        
    # Sort by score desc, then value desc (often total is max value)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    best_val, best_score, best_raw = candidates[0]
    check = _sum_check(text, best_val)
    if check is True:
        best_score = max(best_score, 0.95)  # subtotal + tax equals the chosen total
    elif check is False:
        best_score = min(best_score, 0.4)  # the numbers do not add up: likely a misread
    
    return {"value": best_val, "currency": "USD", "confidence": best_score, "raw": best_raw,
            "sum_check": check} # Default currency USD for now

_NOT_VENDOR = re.compile(
    r"\b(invoice|receipt|bill to|date|due|total|subtotal|tax|thank|terminal|cash|change|"
    r"tel|phone|gst|vat|order|table|qty)\b|\d{3,}|[#@:]", re.I)


def extract_vendor(text: str, tokens: List[Dict]) -> Dict[str, Any]:
    """Vendor = the first of the top lines that does not look like a label, date or number.

    spaCy NER (when the model is installed) only raises confidence for a line that contains an
    organisation entity; it no longer chooses the vendor, because it often picks labels such as
    "Invoice Date" on OCR text.
    """
    lines = [L.strip() for L in text.split('\n') if len(L.strip()) > 2]
    for line in lines[:6]:
        if _NOT_VENDOR.search(line) or sum(c.isalpha() for c in line) < 3:
            continue
        conf = 0.85 if len(line.split()) >= 2 else 0.6
        if nlp and any(e.label_ == "ORG" for e in nlp(line).ents):
            conf = 0.92
        return {"value": line, "confidence": conf, "raw": line}
    if lines:
        return {"value": lines[0], "confidence": 0.3, "raw": lines[0]}
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
