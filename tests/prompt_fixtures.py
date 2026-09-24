from pathlib import Path

from rpera.agents import AGENTS, coordinator_prompt
from rpera.models import SaveNarrativeSettings
from rpera.prompt_store import PromptStore, render_save_narrative_settings


PROMPT_SNAPSHOT = PromptStore(Path(__file__).resolve().parents[1] / "prompts", AGENTS).load(
    frozenset(spec.name for spec in AGENTS.all())
)
PROMPT_TEMPLATES = PROMPT_SNAPSHOT.prompts
TASK_DESCRIPTIONS = PROMPT_SNAPSHOT.task_descriptions
CAPABILITY_DESCRIPTIONS = PROMPT_SNAPSHOT.capability_descriptions
COORDINATOR_PROMPT = coordinator_prompt(PROMPT_TEMPLATES["coordinator"], AGENTS.children())
WORLD_RESEARCHER_PROMPT = PROMPT_TEMPLATES["world_researcher"]
CHARACTER_DESIGNER_PROMPT = PROMPT_TEMPLATES["character_designer"]
LOCATION_DESIGNER_PROMPT = PROMPT_TEMPLATES["location_designer"]
COMPLIANCE_REVIEWER_PROMPT = PROMPT_TEMPLATES["compliance_reviewer"]
EROTIC_OR_NOT_PROMPT = PROMPT_TEMPLATES["EroticOrNot"]
ROLE_PLAYER_PROMPT = PROMPT_TEMPLATES["role_player"]
STYLE_PLANNER_PROMPT = PROMPT_TEMPLATES["style_planner"]
NARRATOR_PROMPT = render_save_narrative_settings(PROMPT_TEMPLATES["narrator"], SaveNarrativeSettings())
CONSISTENCY_CHECKER_PROMPT = render_save_narrative_settings(PROMPT_TEMPLATES["consistency_checker"], SaveNarrativeSettings())
