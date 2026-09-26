#!/usr/bin/env python3
"""
Phase 2: Multi-Representation Preprocessing & Normalization Engine
===================================================================
Produces rich, multi-representation views of business names and addresses:
- Raw, cleaned, and suffix-stripped name variants
- Legal suffix classification and standardization
- Country-aware address standardization (US, India, France)
- Postal code and building/street number extraction
"""

import re
import unicodedata

# ---------------------------------------------------------
# Legal Suffix Dictionary & Regex Patterns
# ---------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "pvtltd": "pvt ltd",
    "limited liability company": "llc",
    "llc": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "limited": "ltd",
    "ltd": "ltd",
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "company": "co",
    "co": "co",
    "societe anonyme": "sa",
    "sa": "sa",
    "sarl": "sarl",
    "gmbh": "gmbh",
    "enterprises": "ent",
    "services": "svc",
}

# Regex to find legal suffix at end of name
_SUFFIX_PATTERN = re.compile(
    r'\b(private\s+limited|pvt\s+limited|pvt\s+ltd|pvtltd|limited\s+liability\s+company|'
    r'limited\s+liability\s+partnership|limited|ltd|corporation|corp|incorporated|inc|'
    r'societe\s+anonyme|sa|sarl|gmbh|llc|llp)\b\.?$',
    re.IGNORECASE
)

# ---------------------------------------------------------
# Address Abbreviation Standardization
# ---------------------------------------------------------
ADDR_ABBREV_MAP = {
    r'\bst\b\.?': 'street',
    r'\brd\b\.?': 'road',
    r'\bave\b\.?': 'avenue',
    r'\bdr\b\.?': 'drive',
    r'\bblvd\b\.?': 'boulevard',
    r'\bln\b\.?': 'lane',
    r'\bct\b\.?': 'court',
    r'\bpl\b\.?': 'place',
    r'\bsq\b\.?': 'square',
    r'\bapt\b\.?': 'apartment',
    r'\bste\b\.?': 'suite',
    r'\bfl\b\.?': 'floor',
    r'\bbldg\b\.?': 'building',
    r'\bhwy\b\.?': 'highway',
    r'\bpkwy\b\.?': 'parkway',
    r'\bopp\b\.?': 'opposite',
    r'\bnr\b\.?': 'near',
    r'\bpo\s+box\b': 'pobox',
    r'\bno\b\.?': 'number',
}

# Precompile address replacements
_ADDR_REGEXES = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDR_ABBREV_MAP.items()]

# Postal code patterns by country
_ZIP_US_REGEX = re.compile(r'\b\d{5}(?:-\d{4})?\b')
_PIN_IN_REGEX = re.compile(r'\b[1-9]\d{5}\b')
_ZIP_FR_REGEX = re.compile(r'\b\d{5}\b')
_HOUSE_NUM_REGEX = re.compile(r'\b\d+[-/]?\w*\b')


def clean_text(text: str) -> str:
    """Normalize unicode, convert to lower case, replace & with and, collapse spaces."""
    if not text or not isinstance(text, str):
        return ""
    # Unicode NFKD
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Replace & with 'and'
    text = text.replace("&", " and ")
    # Replace common separators with spaces
    text = re.sub(r'[\/\\_\-\,\.\:\;\(\)\[\]\{\}\"\'\`\*\+\#\@\!]', ' ', text)
    # Collapse multiple whitespaces
    return re.sub(r'\s+', ' ', text).strip()


def normalize_name(raw_name: str) -> dict:
    """
    Generate multi-representation views for a business name:
    - clean: basic cleaned name
    - base: suffix-stripped core name
    - suffix: standardized suffix code
    - tokens: informative words
    - ngrams: character 3-grams
    """
    clean = clean_text(raw_name)
    if not clean:
        return {
            "clean": "",
            "base": "",
            "suffix": "",
            "tokens": set(),
            "ngrams": set()
        }

    # Match and extract suffix
    suffix_match = _SUFFIX_PATTERN.search(clean)
    suffix = ""
    base = clean
    if suffix_match:
        matched_str = suffix_match.group(1).lower().strip()
        matched_clean = re.sub(r'\s+', ' ', matched_str)
        suffix = LEGAL_SUFFIX_MAP.get(matched_clean, matched_clean)
        base = clean[:suffix_match.start()].strip()

    # Informative tokens (ignore single letters unless digits)
    tokens = {tok for tok in clean.split() if len(tok) > 1 or tok.isdigit()}
    
    # 3-grams of clean name
    padded = f"  {clean}  "
    ngrams = {padded[i:i+3] for i in range(len(padded) - 2)} if len(clean) >= 2 else {clean}

    return {
        "clean": clean,
        "base": base if base else clean,
        "suffix": suffix,
        "tokens": tokens,
        "ngrams": ngrams,
    }


def normalize_address(raw_address: str, country: str = "US") -> dict:
    """
    Country-aware address normalization:
    - Standardize road/street/building abbreviations
    - Extract postal code according to country conventions
    - Extract house / street number
    - Extract address tokens
    """
    clean = clean_text(raw_address)
    if not clean:
        return {
            "clean": "",
            "postal_code": "",
            "street_number": "",
            "tokens": set()
        }

    # Standardize abbreviations
    standardized = clean
    for reg, repl in _ADDR_REGEXES:
        standardized = reg.sub(repl, standardized)
    standardized = re.sub(r'\s+', ' ', standardized).strip()

    # Extract Postal / PIN code based on country
    country_upper = str(country).upper().strip() if country else "US"
    postal_code = ""
    if country_upper in ("INDIA", "IN"):
        m = _PIN_IN_REGEX.search(raw_address)
        if m:
            postal_code = m.group(0)
    elif country_upper in ("FRANCE", "FR"):
        m = _ZIP_FR_REGEX.search(raw_address)
        if m:
            postal_code = m.group(0)
    else:  # Default to US
        m = _ZIP_US_REGEX.search(raw_address)
        if m:
            postal_code = m.group(0)[:5]

    # Extract house or street number (often first digit sequence)
    m_num = _HOUSE_NUM_REGEX.search(standardized)
    street_number = m_num.group(0) if m_num else ""

    tokens = {tok for tok in standardized.split() if len(tok) > 1 or tok.isdigit()}

    return {
        "clean": standardized,
        "postal_code": postal_code,
        "street_number": street_number,
        "tokens": tokens
    }


def preprocess_record(entity_id: str, name: str, address: str, country: str) -> dict:
    """Bundle all normalized components into a single processed record dict."""
    norm_name = normalize_name(name)
    norm_addr = normalize_address(address, country)
    return {
        "entity_id": entity_id,
        "raw_name": name if isinstance(name, str) else "",
        "raw_address": address if isinstance(address, str) else "",
        "country": country if isinstance(country, str) else "US",
        "name_clean": norm_name["clean"],
        "name_base": norm_name["base"],
        "name_suffix": norm_name["suffix"],
        "name_tokens": norm_name["tokens"],
        "name_ngrams": norm_name["ngrams"],
        "addr_clean": norm_addr["clean"],
        "addr_tokens": norm_addr["tokens"],
        "postal_code": norm_addr["postal_code"],
        "street_number": norm_addr["street_number"],
    }


def main():
    print("==================================================================")
    print("PHASE 2: TEXT & ADDRESS NORMALIZATION ENGINE")
    print("==================================================================")
    samples = [
        ("S1-101", "Infosys Technologies Pvt. Ltd.", "Plot No 44, Electronic City, Hosur Road, Bangalore, 560100", "India"),
        ("S1-102", "Amazon.com Services LLC", "410 Terry Ave N, Seattle, WA 98109", "US"),
        ("S1-103", "Societe Generale S.A.", "29 Boulevard Haussmann, 75009 Paris", "France"),
    ]
    for eid, name, addr, country in samples:
        print(f"\n[Record: {eid} ({country})]")
        print(f"  Raw Name:       {name}")
        print(f"  Raw Address:    {addr}")
        rec = preprocess_record(eid, name, addr, country)
        print(f"  Clean Name:     '{rec['name_clean']}'")
        print(f"  Base Name:      '{rec['name_base']}' | Suffix: '{rec['name_suffix']}'")
        print(f"  Clean Address:  '{rec['addr_clean']}'")
        print(f"  Postal / PIN:   '{rec['postal_code']}' | Street Num: '{rec['street_number']}'")
        print(f"  Name Tokens:    {sorted(rec['name_tokens'])}")
    print("\nPhase 2 normalization completed successfully.")


if __name__ == "__main__":
    main()
