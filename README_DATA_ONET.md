**O*NET data downloader**

This repository intentionally does not commit large raw data files. Use the included script to download specific O*NET content model files into `datasets/onet/` on your machine.

Usage:

```bash
python scripts/download_onet.py --out datasets/onet \
  --categories "Skills" "Knowledge" "Education" "Work Activities" "Occupation" \
  "Abilities" "Work Styles" "Experience and Training" "Occupational Outlook"
```

Notes:
- The script scrapes the O*NET content page for downloadable CSV/ZIP links and saves matching CSVs.
- The `datasets/` directory is ignored by `.gitignore` to avoid committing raw data; keep datasets local or store in external object storage (S3, GCS) for collaboration.
- If the site structure changes or the script can't find files, provide direct file URLs instead of the page URL.
