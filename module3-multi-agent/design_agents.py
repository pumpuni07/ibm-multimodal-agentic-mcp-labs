"""
M3L1 — Design Specialized Agents for a Recommendation System
============================================================
Defines the six specialized agents of the restaurant/recipe recommendation
system, each with a role, goal, and backstory, plus their task
specifications with expected outputs.

Reconstructed to the lab's summary and grading requirements
(M3L1_food_style_expert_goal: the Food Style Expert configuration showing
role, goal, and backstory).

Agents: User Profile Generator · RAG Retriever · Food Trend Analyst ·
Food Style Expert · Nutrition Expert · Recommendation Expert
"""

import os

from crewai import Agent, Task
from crewai.llm import LLM

WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
MODEL_ID = os.environ.get("M3_AGENT_MODEL", "watsonx/ibm/granite-3-3-8b-instruct")


def make_llm() -> LLM:
    """Shared LLM handle for all six agents (credentials from environment)."""
    return LLM(
        model=MODEL_ID,
        base_url=WATSONX_URL,
        api_key=os.environ.get("WATSONX_APIKEY"),
        temperature=0.3,
    )


def build_user_profile_generator() -> Agent:
    return Agent(
        role="User Profile Generator",
        goal=(
            "Analyze a user's review history, ratings, and stated preferences "
            "to produce a structured profile of their tastes, dietary "
            "constraints, and dining habits."
        ),
        backstory=(
            "A behavioral data analyst who has spent years turning messy "
            "customer histories into crisp preference profiles, known for "
            "never over-interpreting sparse data."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_rag_retriever() -> Agent:
    return Agent(
        role="RAG Retriever",
        goal=(
            "Query the multimodal vector databases for restaurants and "
            "recipes most relevant to the user's profile, returning "
            "candidates with their metadata and similarity context."
        ),
        backstory=(
            "A retrieval specialist fluent in embeddings and metadata "
            "filtering, obsessive about surfacing candidates that are "
            "actually grounded in the indexed data rather than guessed."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_food_trend_analyst() -> Agent:
    return Agent(
        role="Food Trend Analyst",
        goal=(
            "Identify which current food trends are relevant to the "
            "candidate restaurants and recipes, and assess how trendiness "
            "should influence the recommendation."
        ),
        backstory=(
            "A culinary journalist turned analyst who tracks food movements "
            "across cities and social platforms, skilled at separating "
            "durable trends from passing fads."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_food_style_expert() -> Agent:
    """The agent whose configuration is the M3L1 screenshot requirement."""
    return Agent(
        role="Food Style Expert",
        goal=(
            "Analyze the cuisines, cooking techniques, and flavor profiles "
            "of the candidate restaurants and recipes, and evaluate how well "
            "each matches the user's taste profile."
        ),
        backstory=(
            "A classically trained chef and food writer with two decades of "
            "experience across regional cuisines, celebrated for describing "
            "flavor profiles precisely and matching dishes to palates with "
            "uncanny accuracy."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_nutrition_expert() -> Agent:
    return Agent(
        role="Nutrition Expert",
        goal=(
            "Evaluate the nutritional content and dietary fit of each "
            "candidate against the user's constraints, flagging conflicts "
            "such as allergens or dietary-pattern mismatches."
        ),
        backstory=(
            "A registered dietitian who bridges clinical nutrition and "
            "everyday eating, pragmatic about balance rather than "
            "perfection."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_recommendation_expert() -> Agent:
    return Agent(
        role="Recommendation Expert",
        goal=(
            "Synthesize the profile, retrieval results, trend analysis, "
            "style analysis, and nutrition evaluation into a final ranked "
            "recommendation with clear reasoning for each pick."
        ),
        backstory=(
            "A former concierge and recommender-systems engineer who "
            "believes a great recommendation must be both personally "
            "resonant and honestly justified."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_all_agents() -> dict[str, Agent]:
    """All six agents keyed by role, in workflow order."""
    agents = [
        build_user_profile_generator(),
        build_rag_retriever(),
        build_food_trend_analyst(),
        build_food_style_expert(),
        build_nutrition_expert(),
        build_recommendation_expert(),
    ]
    return {agent.role: agent for agent in agents}


def build_tasks(agents: dict[str, Agent]) -> list[Task]:
    """Task specifications with expected outputs, one per agent."""
    profile_task = Task(
        description=(
            "Analyze the user data provided in {user_data} and extract their "
            "cuisine preferences, dietary constraints, price sensitivity, "
            "and ambiance preferences."
        ),
        expected_output="A structured user profile with the four preference categories.",
        agent=agents["User Profile Generator"],
    )
    retrieval_task = Task(
        description=(
            "Using the user profile, query the restaurant and recipe vector "
            "databases and return the top candidate matches with metadata."
        ),
        expected_output="A list of 5-10 candidates with names, cuisines, and metadata.",
        agent=agents["RAG Retriever"],
        context=[profile_task],
    )
    trend_task = Task(
        description="Assess which current food trends apply to the candidates.",
        expected_output="A trend-relevance note for each candidate.",
        agent=agents["Food Trend Analyst"],
        context=[retrieval_task],
    )
    style_task = Task(
        description=(
            "Analyze each candidate's cuisine, techniques, and flavor "
            "profile against the user's taste profile."
        ),
        expected_output="A style-match assessment per candidate with flavor notes.",
        agent=agents["Food Style Expert"],
        context=[retrieval_task],
    )
    nutrition_task = Task(
        description=(
            "Evaluate each candidate's nutritional fit against the user's "
            "dietary constraints, flagging any conflicts."
        ),
        expected_output="A dietary-fit verdict per candidate with flagged conflicts.",
        agent=agents["Nutrition Expert"],
        context=[retrieval_task],
    )
    recommendation_task = Task(
        description=(
            "Synthesize all analyses into a final ranked list of two or "
            "three recommendations with reasoning."
        ),
        expected_output="A ranked recommendation list with one-paragraph justification each.",
        agent=agents["Recommendation Expert"],
        context=[trend_task, style_task, nutrition_task],
    )
    return [profile_task, retrieval_task, trend_task, style_task,
            nutrition_task, recommendation_task]


if __name__ == "__main__":
    agents = build_all_agents()
    tasks = build_tasks(agents)
    expert = agents["Food Style Expert"]
    print("Food Style Expert configuration:")
    print(f"  role: {expert.role}")
    print(f"  goal: {expert.goal}")
    print(f"  backstory: {expert.backstory}")
    print(f"\n{len(agents)} agents and {len(tasks)} tasks defined.")
