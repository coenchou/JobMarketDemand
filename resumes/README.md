# Resume parsing

This folder is intended for plain-text resume files (.txt).

Run the parser on one resume with:

```sh
/Users/coenchou/JobMarketDemand/venv/bin/python scripts/resume_parser.py --resume resumes/resume_1_data_analyst.txt
```

The parser currently returns:
- detected skills
- education details
- experience bullets
- matching ONET vocabulary terms from the datasets folder
