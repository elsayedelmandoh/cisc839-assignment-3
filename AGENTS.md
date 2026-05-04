---
name: agents
description: >
  you are senior ai engineer act as ai assistant
---

# style + formatting rules
 
- all output (code, markdown, notebooks, docs) must be written in **lowercase**
- never use the em dash character `—` anywhere, under any circumstances
- use plain hyphens `-` for ranges and alternatives
- keep prose tight: no filler, no hedging, no fluff

# safety before action
 
- never delete anything without explicit user confirmation.
- always show first what you plan to change, then act implementation

## rules
1. all content: lowercase for consistency
2. preserve everything - do not delete files or content unless explicitly asked
3. always ask for confirmation before making structural changes
4. show what you found before modifying anything
5. never hardcode secrets. use .env and environment variables only by settings.py file

## development workflow
- write new logic in src/ as reusable modules
- use notebooks/ for exploration and prototyping
- move stable code from notebooks to src/
- keep tests deterministic and repeatable
- tests: tests/
- docs: docs/

## notebook + code conventions
- all notebooks must import every required library in the **first code cell** - no mid-notebook imports
- all reusable functions go into `src/utils/helpers.py` - notebooks call them, not redefine them
- all constants, paths, config variables go into `src/config/settings.py` - notebooks import from there
- no hardcoded paths, thresholds, or magic numbers in notebooks - use settings.py constants

## important paths
- config for all variables: src/config/settings.py
- utils for all functions: src/utils/helper.py
- architecture: docs/01-project-definition/07-architecture.md
- stack details: docs/01-project-definition/06-stack.md
