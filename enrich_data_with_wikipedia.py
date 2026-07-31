"""
Wikipedia Enrichment Pipeline
Global Startup Intelligence Graph

Enriches 6000 YC company files with Wikipedia content.
Uses GPT-4.1 via Azure OpenAI to verify relevance of Wikipedia pages.
Original files in input/ are left untouched.
Enriched copies are written to input_enriched/.
Tracks progress so it can resume if interrupted.
"""

import requests
import os
import time
import json
import re
import shutil
from pathlib import Path
from azure.identity import DefaultAzureCredential

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR = "input"
OUTPUT_DIR = "input_enriched"
PROGRESS_FILE = "enrichment_progress.json"
LOG_FILE = "enrichment_log.txt"
WIKIPEDIA_HEADERS = {"User-Agent": "StartupIntelligenceGraph/1.0 (kamal@example.com)"}
RATE_LIMIT_SECONDS = 0.5        # be respectful to Wikipedia
MIN_CONTENT_LENGTH = 300        # ignore very short/stub pages

# Azure OpenAI settings — must match your settings.yaml
AZURE_OPENAI_ENDPOINT = "https://aifoundry-allpurposes.cognitiveservices.azure.com"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# ── Azure Authentication ───────────────────────────────────────────────────────

credential = DefaultAzureCredential()

def get_azure_token():
    """Get Azure AD token for Azure OpenAI using az login credentials."""
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token

# ── Progress Tracking ──────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "enriched": [],       # successfully enriched
        "not_found": [],      # no Wikipedia page found — copied as-is
        "irrelevant": [],     # page found but LLM said not relevant — copied as-is
        "failed": [],         # errors during processing
        "already_done": []    # already in output folder from previous run
    }

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ── Company Name Extraction ────────────────────────────────────────────────────

def extract_company_name(filepath):
    """Extract company name from YC profile file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "YC Startup Profile:" in line:
                    return line.split("YC Startup Profile:")[-1].strip()
        return Path(filepath).stem.replace("-", " ").replace("_", " ").title()
    except Exception:
        return Path(filepath).stem

def is_already_in_output(filepath):
    """Check if file already exists in output folder."""
    output_path = Path(OUTPUT_DIR) / filepath.name
    return output_path.exists()

# ── Wikipedia API ──────────────────────────────────────────────────────────────

def search_wikipedia(company_name):
    """Search Wikipedia for the company and return best matching title."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{company_name} company",
        "srlimit": 3,
        "format": "json"
    }
    try:
        response = requests.get(url, headers=WIKIPEDIA_HEADERS, params=params, timeout=10)
        results = response.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
        return None
    except Exception as e:
        log(f"  Search error for {company_name}: {e}")
        return None

def get_wikipedia_content(title):
    """Fetch full plain text content of a Wikipedia page."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json"
    }
    try:
        response = requests.get(url, headers=WIKIPEDIA_HEADERS, params=params, timeout=10)
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return page.get("extract", None)
    except Exception as e:
        log(f"  Content fetch error for {title}: {e}")
        return None

# ── LLM Relevance Check ────────────────────────────────────────────────────────

def is_relevant(content, company_name):
    """Use GPT-4.1 to verify Wikipedia content is about the correct startup."""
    if not content or len(content) < MIN_CONTENT_LENGTH:
        return False

    snippet = content[:500]

    prompt = f"""You are verifying whether a Wikipedia article is about a specific startup company.

Company name: {company_name}

Wikipedia article snippet:
{snippet}

Is this Wikipedia article about the startup company called "{company_name}"?
Answer with only YES or NO."""

    try:
        token = get_azure_token()
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=15
        )
        answer = response.json()["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        log(f"  LLM relevance check failed for {company_name}: {e}")
        return company_name.lower() in content.lower()

# ── Alternate Name Generation ──────────────────────────────────────────────────

def get_alternate_names(company_name):
    """Generate alternate search names to try if first attempt fails."""
    alternates = [
        company_name + " startup",
        company_name + " Inc",
        company_name + " Inc.",
        re.sub(r'\s+(Inc\.?|LLC|Ltd\.?|Corp\.?)$', '', company_name, flags=re.IGNORECASE)
    ]
    return [a for a in alternates if a != company_name]

# ── Enrichment ─────────────────────────────────────────────────────────────────

def enrich_file(source_path, company_name):
    """
    Copy source file to output folder and append Wikipedia content if found.
    Original file is never modified.
    Returns: 'enriched', 'not_found', 'irrelevant', or 'failed'
    """
    output_path = Path(OUTPUT_DIR) / source_path.name

    try:
        # Always copy original file to output folder first
        shutil.copy2(source_path, output_path)

        names_to_try = [company_name] + get_alternate_names(company_name)

        for name in names_to_try:
            title = search_wikipedia(name)
            if not title:
                continue

            content = get_wikipedia_content(title)
            if not content:
                continue

            if is_relevant(content, company_name):
                # Append Wikipedia content to the copy only
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- WIKIPEDIA ENRICHMENT ---\n")
                    f.write(f"Source: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}\n\n")
                    f.write(content)
                return "enriched"
            else:
                log(f"  Irrelevant page found: '{title}' for '{company_name}'")

        # No Wikipedia content found — copy exists but without enrichment
        return "not_found"

    except Exception as e:
        log(f"  Failed to enrich {company_name}: {e}")
        return "failed"

# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_enrichment():
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    progress = load_progress()
    processed = set(
        progress["enriched"] +
        progress["not_found"] +
        progress["irrelevant"] +
        progress["failed"] +
        progress["already_done"]
    )

    files = sorted(Path(INPUT_DIR).glob("*.txt"))
    total = len(files)

    log(f"\n{'='*60}")
    log(f"Wikipedia Enrichment Pipeline")
    log(f"Input:  {INPUT_DIR}/")
    log(f"Output: {OUTPUT_DIR}/")
    log(f"Total files: {total}")
    log(f"Already processed: {len(processed)}")
    log(f"Remaining: {total - len(processed)}")
    log(f"{'='*60}\n")

    for i, filepath in enumerate(files):
        filename = filepath.name

        if filename in processed:
            print(f"[{i+1}/{total}] Skipping: {filename}")
            continue

        if is_already_in_output(filepath):
            progress["already_done"].append(filename)
            save_progress(progress)
            print(f"[{i+1}/{total}] Already in output: {filename}")
            continue

        company_name = extract_company_name(filepath)
        print(f"[{i+1}/{total}] Processing: {company_name}")

        result = enrich_file(filepath, company_name)
        progress[result].append(filename)
        save_progress(progress)

        if result == "enriched":
            log(f"  ✓ Enriched: {company_name}")
        elif result == "not_found":
            log(f"  ✗ Not found (copied as-is): {company_name}")
        elif result == "irrelevant":
            log(f"  ~ Irrelevant (copied as-is): {company_name}")
        elif result == "failed":
            log(f"  ! Failed: {company_name}")

        time.sleep(RATE_LIMIT_SECONDS)

    log(f"\n{'='*60}")
    log(f"Enrichment Complete!")
    log(f"✓ Enriched:     {len(progress['enriched'])}")
    log(f"✗ Not found:    {len(progress['not_found'])}")
    log(f"~ Irrelevant:   {len(progress['irrelevant'])}")
    log(f"! Failed:       {len(progress['failed'])}")
    log(f"- Already done: {len(progress['already_done'])}")
    log(f"{'='*60}\n")
    log(f"Original files untouched in: {INPUT_DIR}/")
    log(f"Enriched files written to:   {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_enrichment()