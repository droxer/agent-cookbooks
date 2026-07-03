"""Agent Skills: folder-based expertise with progressive disclosure.

Skills (introduced for Claude in 2025, now an open pattern) package procedural
knowledge - how to review a PR, how to write a postmortem - as folders of
instructions the agent loads *on demand*. The design principle is progressive
disclosure, the same one behind ``examples/tools/code_execution.py``:

1. **At startup** only each skill's frontmatter (name + one-line description)
   enters the system prompt - a few dozen tokens per skill, so an agent can
   carry hundreds of skills.
2. **On use** the agent calls ``use_skill`` and receives the full SKILL.md
   body - the detailed procedure - only for the skill the task needs.
3. **Skills are data, not code** - adding expertise means dropping a folder
   into ``library/``, no agent changes, no redeploy. Domain experts can write
   them.

Each skill lives at ``library/<skill-name>/SKILL.md`` with YAML-style
frontmatter (``name``, ``description``) followed by markdown instructions.
Real skill folders can also carry scripts and reference files alongside
SKILL.md; the agent reads those just-in-time too.

Run with:
    python examples/skills/skills_agent.py
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from examples.harness.agent_harness import AgentHarness, HarnessConfig

load_dotenv()

SKILLS_ROOT = Path(__file__).parent / "library"


# ---------------------------------------------------------------------------
# Skill discovery: parse only the frontmatter at startup
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def instructions(self) -> str:
        """Load the full SKILL.md body - only called when the skill is used."""
        _, body = parse_skill_file(self.path.read_text())
        return body


def parse_skill_file(text: str) -> tuple[dict, str]:
    """Split SKILL.md into frontmatter metadata and the instruction body.

    Frontmatter is the block between the leading '---' lines, as simple
    'key: value' pairs (no YAML dependency needed for flat metadata).
    """
    if not text.startswith("---"):
        return {}, text
    _, frontmatter, body = text.split("---", 2)
    meta = {}
    for line in frontmatter.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def discover_skills(root: Path = SKILLS_ROOT) -> dict[str, Skill]:
    """Scan the library and read ONLY the metadata of each skill."""
    skills = {}
    for skill_file in sorted(root.glob("*/SKILL.md")):
        meta, _ = parse_skill_file(skill_file.read_text())
        name = meta.get("name", skill_file.parent.name)
        skills[name] = Skill(
            name=name,
            description=meta.get("description", "(no description)"),
            path=skill_file,
        )
    return skills


SKILLS = discover_skills()


@tool
def use_skill(name: str) -> str:
    """Load the full instructions for a skill by name. Call this BEFORE doing
    a task a skill covers, then follow the instructions exactly."""
    skill = SKILLS.get(name)
    if skill is None:
        return f"Unknown skill {name!r}. Available: {sorted(SKILLS)}"
    return f"# Skill: {skill.name}\n\n{skill.instructions()}"


def build_system_prompt() -> str:
    # Progressive disclosure: the prompt carries one line per skill, not the
    # full instructions.
    catalog = "\n".join(f"- {s.name}: {s.description}" for s in SKILLS.values())
    return f"""You are a software engineering assistant.

You have access to skills - packaged procedures for specific tasks:

{catalog}

When a task matches a skill's description, you MUST call use_skill to load it
and follow its instructions before answering. If no skill matches, answer
normally."""


def main():
    llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
    harness = AgentHarness(model=llm, tools=[use_skill],
                           config=HarnessConfig(max_turns=4))

    result = harness.run(
        system_prompt=build_system_prompt(),
        user_input=(
            "Write up yesterday's incident: 14:02 checkout latency alarms, "
            "14:10 on-call paged, 14:25 traced to connection pool exhaustion "
            "after the 13:55 deploy, 14:31 rollback started, 14:40 recovered. "
            "~18% of checkout requests failed during the window."
        ),
    )

    print("\n=== RESULT ===")
    print(f"stop_reason: {result.stop_reason.value}")
    print(result.final_answer)


if __name__ == "__main__":
    main()
