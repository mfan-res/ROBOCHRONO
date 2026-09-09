---
name: Bug report
about: Something behaves wrongly or fails
labels: bug
---

**Environment**
- OS / distribution:
- GPU(s) and driver version (if local models are involved):
- Python version and environment (`tf4` / `tf5` / api):
- Commit (`git rev-parse --short HEAD`):

**Command that reproduces it**

```bash
# the exact command line, including --suite / --models / --limit-items
```

**Preflight output**

```
# paste the full output of:
#   .venvs/tf4/bin/python -m robochrono preflight --models <slug>
```

**What happened, and what you expected**

Include the relevant log lines (`error:` lines, tracebacks, or the
report.md flags). If the problem is a score rather than a crash, say which
run directory and which (scenario, dimension) cell.
