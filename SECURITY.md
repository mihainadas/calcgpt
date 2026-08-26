# Security policy

## Supported versions

CalcGPT is currently developed on the default branch and does not yet publish maintained release lines.

| Version | Supported |
|---|---|
| Default branch | Yes |
| Historical snapshots | No |

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting page for this repository. If that channel is unavailable, contact the maintainer through the contact information on their GitHub profile. Do not disclose a suspected vulnerability in a public issue, discussion, or pull request.

Include the affected revision, impact, reproduction steps, and any suggested mitigation. Remove credentials and unrelated personal information from logs. Receipt and remediation timing depend on maintainer availability; no fixed response deadline is promised.

## Scope notes

CalcGPT loads local model and dataset files and uses third-party machine-learning dependencies. Only load artifacts from sources you trust. A model file is not a security boundary, and generated arithmetic output must not be used as authorization, financial, safety, or integrity-critical input.
