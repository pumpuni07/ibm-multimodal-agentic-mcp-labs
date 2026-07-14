"""Tests for M3L1. Real CrewAI Agent/Task objects; no LLM calls occur at
construction, so nothing is mocked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design_agents
from crewai import Agent, Task

EXPECTED_ROLES = [
    "User Profile Generator", "RAG Retriever", "Food Trend Analyst",
    "Food Style Expert", "Nutrition Expert", "Recommendation Expert",
]


def test_all_six_agents_are_built_with_expected_roles():
    agents = design_agents.build_all_agents()
    assert list(agents.keys()) == EXPECTED_ROLES
    assert all(isinstance(a, Agent) for a in agents.values())


def test_every_agent_has_nonempty_role_goal_backstory():
    for agent in design_agents.build_all_agents().values():
        assert agent.role.strip()
        assert len(agent.goal.strip()) > 40
        assert len(agent.backstory.strip()) > 40


def test_food_style_expert_screenshot_fields():
    expert = design_agents.build_food_style_expert()
    assert expert.role == "Food Style Expert"
    assert "flavor profiles" in expert.goal
    assert "cuisines" in expert.goal
    assert "chef" in expert.backstory


def test_agents_do_not_delegate():
    for agent in design_agents.build_all_agents().values():
        assert agent.allow_delegation is False


def test_six_tasks_one_per_agent_in_workflow_order():
    agents = design_agents.build_all_agents()
    tasks = design_agents.build_tasks(agents)
    assert len(tasks) == 6
    assert all(isinstance(t, Task) for t in tasks)
    assert [t.agent.role for t in tasks] == EXPECTED_ROLES


def test_task_context_wiring_matches_hybrid_workflow():
    agents = design_agents.build_all_agents()
    tasks = design_agents.build_tasks(agents)
    profile, retrieval, trend, style, nutrition, recommendation = tasks
    assert retrieval.context == [profile]
    assert trend.context == [retrieval]
    assert style.context == [retrieval]
    assert nutrition.context == [retrieval]
    assert recommendation.context == [trend, style, nutrition]


def test_every_task_declares_expected_output():
    agents = design_agents.build_all_agents()
    for task in design_agents.build_tasks(agents):
        assert task.expected_output.strip()


def test_credentials_from_environment_only():
    source = Path(design_agents.__file__).read_text()
    assert "os.environ" in source
    assert "sk-" not in source
