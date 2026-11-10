# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## IMPORTANT: Code Rules (MUST follow)

### Code Style
- **Imports:** ALWAYS at the TOP of the file. NEVER inside functions/methods.
- **Keep it Simple:** Avoid over-engineering. Minimal code for the task.
- **Dicts:** Comment key→value types if not obvious, e.g. `# lemma -> frequency`

### Error Handling
- **Fail Fast:** Let errors crash. NO unnecessary try/except.
- **Warnings over Catches:** Prefer logging warnings over catching errors.

### Logging
- **Sparse:** Only log what's essential for understanding the flow.

---

## Project Overview

See [README.md](README.md) for what the pipeline does, its stages, outputs and project structure.
Configuration lives in `config.yaml`.

## Development Commands

The project uses the conda environment `LanguageApp-new` (see `environment.yml`).

```bash
# Run the pipeline
~/anaconda3/bin/conda run -n LanguageApp-new python3 main.py

# Run a test / evaluation script (run from the project root)
PYTHONPATH=.:test ~/anaconda3/bin/conda run -n LanguageApp-new python3 test/test_post_processor.py

# Run inline Python with project imports
~/anaconda3/bin/conda run -n LanguageApp-new python3 -c "
import sys
sys.path.insert(0, '.')
from src.vocab import Vocabulary
"
```

LLM steps need `GOOGLE_API_KEY` (or `OPENAI_API_KEY`) in `.env`. Corpus files go under `data/` (not in the repository).
