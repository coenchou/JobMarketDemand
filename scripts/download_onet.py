#!/usr/bin/env python3
"""Download selected O*NET Content Model files and extract CSVs into datasets/onet/

Usage:
  python scripts/download_onet.py --out datasets/onet --categories Skills Knowledge Education "Work Activities" Occupation

Notes:
- The script scrapes the provided O*NET content page for downloadable CSV/ZIP links,
  downloads matching files, and saves them under the output directory.
- Data files are intentionally written to `datasets/onet/` which is ignored by .gitignore.
"""
import argparse
import os
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from zipfile import ZipFile

DEFAULT_URL = "https://www.onetcenter.org/content.html#cm1"

CATEGORY_KEYWORDS = {
    "skills": ["skill", "skills"],
    "knowledge": ["knowledge"],
    "education": ["education"],
    "work activities": ["work_activities", "workactivities", "work-activities", "work_activities"],
    "occupation": ["occupation", "occupational", "occupation-title", "occupation_title"],
    "abilities": ["ability", "abilities"],
    "work styles": ["work_style", "work-styles", "workstyles"],
    "experience and training": ["experience", "training"],
    "occupational outlook": ["occupational_outlook", "occupational-outlook", "occupationaloutlook"]
}


def find_download_links(page_url):
    resp = requests.get(page_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith((".zip", ".csv", ".xlsx")):
            full = urljoin(page_url, href)
            links.append(full)
    return links


def download_url(url):
    print(f"Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def save_file(path, content_bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content_bytes)


def match_filename(name, wanted_categories):
    name_lower = name.lower()
    for cat in wanted_categories:
        keywords = CATEGORY_KEYWORDS.get(cat.lower(), [cat.lower()])
        for kw in keywords:
            if kw in name_lower:
                return True
    return False


def extract_zip_and_save(content_bytes, out_dir, wanted_categories):
    z = ZipFile(BytesIO(content_bytes))
    saved = []
    for member in z.namelist():
        base = os.path.basename(member)
        if not base:
            continue
        if not re.search(r"\.(csv|xlsx|xls)$", base, re.IGNORECASE):
            continue
        if match_filename(base, wanted_categories):
            target = os.path.join(out_dir, base)
            print(f"Extracting {base} -> {target}")
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            saved.append(target)
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL, help="O*NET content page URL")
    p.add_argument("--out", default="datasets/onet", help="Output directory")
    p.add_argument("--categories", nargs="+", required=True, help="Categories to download (e.g. Skills Knowledge Education)")
    args = p.parse_args()

    out_dir = args.out
    wanted = [c.lower() for c in args.categories]
    print(f"Target categories: {wanted}")
    print(f"Scraping links from {args.url} ...")
    links = find_download_links(args.url)
    if not links:
        print("No downloadable links found on the page. You may need to provide a direct file URL.")
        sys.exit(1)

    saved_files = []
    for link in links:
        parsed = urlparse(link)
        fname = os.path.basename(parsed.path)
        try:
            data = download_url(link)
        except Exception as e:
            print(f"Failed to download {link}: {e}")
            continue

        if fname.lower().endswith(".zip"):
            saved = extract_zip_and_save(data, out_dir, wanted)
            saved_files.extend(saved)
        elif fname.lower().endswith((".csv", ".xlsx", ".xls")):
            if match_filename(fname, wanted):
                target = os.path.join(out_dir, fname)
                print(f"Saving {os.path.splitext(fname)[1].upper()} {fname} -> {target}")
                save_file(target, data)
                saved_files.append(target)
        else:
            # try to inspect content-type or filename inside
            print(f"Skipping unknown file type: {fname}")

    if saved_files:
        print("\nDownloaded and saved the following files:")
        for s in saved_files:
            print(" - ", s)
    else:
        print("No matching files were saved. Try using different category names or a direct file URL.")


if __name__ == "__main__":
    main()
