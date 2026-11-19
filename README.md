# RideOps L1 Escalation Assistant

## Overview

A working v0 for RideOps L1 Support Agents.

The tool uses historical support tickets to help agents:
- find similar past issues
- understand historical escalation patterns
- see Engineering outcomes
- assess whether additional L1 checks are warranted
- generate a structured Engineering handoff

## User

L1 Support Agent

## How to run

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

   python -m pip install -r requirements.txt

4. Run:

   streamlit run app.py

## Approach

Historical ticket similarity is calculated using TF-IDF and cosine similarity.

The tool does not automatically decide whether a ticket should be escalated. It provides evidence so the L1 agent can make the decision.

## Why this approach

TF-IDF was chosen because the supplied dataset contains only 1,444 tickets and the first version needed to be simple, deterministic, fast and explainable.

## Limitations

The dataset does not contain live operational diagnostics, so the tool does not attempt to perform root-cause diagnosis.

Weak historical matches are explicitly surfaced rather than treated as reliable evidence.

## Validation

The v0 was tested on recurring issues including:
- Roster / booking not syncing
- Live tracking / GPS
- Report export
- How-to / feature request
- weak-evidence cases