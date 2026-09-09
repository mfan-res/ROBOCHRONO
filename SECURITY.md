# Security

## Reporting

Report suspected vulnerabilities privately to **mfan@gimai.io**. Please do
not open a public issue for something exploitable before it is fixed.
You can expect an acknowledgement within seven days.

## What counts

Anything where this code processes input it should distrust:

- archive extraction in `tools/download_data.py` (path traversal and
  similar);
- handling of API credentials — keys are read from environment variables
  and must never land in configurations, results or logs;
- parsing of model output and of downloaded dataset files.

Evaluation results being wrong is a bug, not a vulnerability — open a
regular issue for those.
