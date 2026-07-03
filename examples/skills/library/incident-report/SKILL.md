---
name: incident-report
description: Write a production incident report (postmortem) from a timeline of events. Use when asked to summarize or write up an outage or incident.
---

# Incident Report

Produce a blameless postmortem from the raw timeline the user provides.

## Structure

1. **Summary** - two sentences: user-visible impact, duration, root cause.
2. **Impact** - who/what was affected, quantified (requests failed, minutes down).
3. **Timeline** - normalized to UTC, one line per event, detection and
   mitigation moments called out.
4. **Root cause** - the causal chain, not just the trigger. Ask "why" until
   you reach a process or design decision.
5. **Action items** - each with an owner category (detection, prevention,
   mitigation) and a concrete, verifiable done-condition.

## Rules

- Blameless: name systems and decisions, never individuals.
- Never speculate: mark unknowns as "under investigation" rather than guessing.
