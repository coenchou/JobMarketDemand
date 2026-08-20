"""
Presentable skill names.

O*NET stores "Amazon Web Services AWS SageMaker" and "Ansible software"; no
candidate writes that on a resume and no reader wants to read it in a report.
This strips vendor prefixes and category suffixes down to what the skill is
actually called, without touching the canonical name the scoring code matches
on.
"""

from __future__ import annotations

from typing import Dict, List

from src.embeddings import _norm

_VENDOR_PREFIXES: List[List[str]] = [
    ["amazon", "web", "services"], ["amazon", "web", "service"],
    ["grafana", "labs"], ["hewlett", "packard", "enterprise"],
    ["hewlett", "packard"], ["red", "hat"], ["the", "mathworks"],
    ["national", "instruments"], ["qlik", "tech"], ["ibm"], ["microsoft"],
    ["oracle"], ["google"], ["amazon"], ["adobe"], ["apache"], ["esri"],
    ["sap"], ["cisco"], ["intel"], ["nvidia"], ["jetbrains"], ["atlassian"],
    ["mathworks"], ["autodesk"], ["siemens"], ["dassault"], ["ptc"],
    ["altair"], ["ansys"], ["wolfram"], ["minitab"], ["statacorp"],
    ["mozilla"], ["vmware"], ["citrix"], ["corel"], ["intuit"], ["tibco"],
    ["splunk"], ["elastic"], ["hashicorp"], ["salesforce", "com"],
]

_SUFFIXES = {"software", "systems", "system", "tools", "tool", "suite",
             "application", "applications", "platform", "program"}

_OVERRIDES: Dict[str, str] = {
    "structured query language sql": "SQL",
    "extensible markup language xml": "XML",
    "hypertext markup language html": "HTML",
    "cascading style sheets css": "CSS",
    "javascript object notation json": "JSON",
    "bidirectional encoder representations from transformers bert": "BERT",
    "oracle java 2 platform enterprise edition j2ee": "Java EE",
    "amazon web services aws software": "AWS",
    "microsoft office software": "Microsoft Office",
    "unix shell": "Unix shell",
    "shell script": "Shell scripting",
}


def pretty_skill(name: str) -> str:
    """
    Human-facing form of a tool name. Falls back to the original whenever
    stripping would leave nothing meaningful.
    """
    if not name:
        return ""
    key = _norm(name)
    if key in _OVERRIDES:
        return _OVERRIDES[key]

    words = name.split()
    lowered = [w.lower().strip(".,") for w in words]

    for prefix in _VENDOR_PREFIXES:
        n = len(prefix)
        if len(words) > n and lowered[:n] == prefix:
            words, lowered = words[n:], lowered[n:]
            break

    while len(words) > 1 and lowered[-1] in _SUFFIXES:
        words, lowered = words[:-1], lowered[:-1]

    return " ".join(words) if words else name


def pretty_all(names: List[str]) -> List[str]:
    return [pretty_skill(n) for n in names]
