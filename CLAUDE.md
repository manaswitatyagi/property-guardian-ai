# Project Context

An Agentic Property Intelligence System for Continuous Risk Monitoring and Loss Prevention


# Vision

Property Guardian AI transforms property insurance from periodic underwriting into continuous risk monitoring.

Traditional insurance evaluates risk once per year. Property risk, however, changes continuously because of:

Weather
Flood exposure
Climate trends
Property aging
Inflation
Neighborhood changes

The objective is to create a multi-agent system that continuously observes properties, detects changes in risk, explains why risk changed, and recommends preventive actions.


The system should answer:

Why did this property's risk increase?
What are the primary risk drivers?
What actions can reduce losses?
How has risk evolved over time?

The system would be ideally used by Underwriters. Here's what I understand about them and the insurance industry. 

The explainability gap is brutal when regulators or auditors want to know why a decision was made. Traditional models with clear feature weights and thresholds are way easier to defend than black box neural networks.

For explainability, SHAP values and feature importance help but they're still approximations. What actually satisfies auditors is documenting your model validation process, bias testing, and having clear override procedures when the model makes questionable calls.

They need to see the model perform consistently over months before they'll fully buy it. So,   shadow scoring where AI runs alongside manual decisons is a game changer. Track where they agree and disagree.

Regulatory compliance requires model documentation showing you tested for bias, validated on holdout data, and monitor performance drift. Most regulators are fine with AI if you can demonstrate proper governanc

# Guiding Principles

Never output a score without an explanation.

Every risk score should include:

Main contributors
Supporting evidence
Confidence level
Recommendations

Risk prediction can rely on:

XGBoost
Random Forest
SHAP explanations


# Agentic Design

Each agent should have:

A clear responsibility
Well-defined inputs
Structured outputs
Minimal coupling

Agents communicate through structured objects.



# Architecture

User

↓

Property Guardian AI

↓

Orchestrator Agent

├── Property Agent
├── Weather Agent
├── Flood Agent
├── Climate Agent
├── Neighborhood Agent
├── Inflation Agent
├── Recommendation Agent
└── Memory Agent

↓

Risk Reasoning Agent

↓

Dashboard / API


# Agents

Property Agent

Purpose:

Build a property profile.

Inputs:

Address
Latitude
Longitude

Features:

Year built
Property age
Estimated roof age
Home value
Square footage

Output:

PropertyProfile
Weather Agent

Purpose:

Measure weather exposure.

Data Sources:

NOAA Storm Events

Features:

Hail frequency
Wind events
Heavy rainfall events
Extreme temperatures

Output:

WeatherRisk
Flood Agent

Purpose:

Evaluate flood exposure.

Data Sources:

FEMA

Features:

Flood zone
Historical flooding
Distance to water

Output:

FloodRisk
Climate Agent

Purpose:

Monitor long-term climate trends.

Data Sources:

NOAA
NASA

Features:

Heat trends
Precipitation trends
Wildfire trends

Output:

ClimateRisk
Neighborhood Agent

Purpose:

Evaluate local conditions.

Data Sources:

Census
OpenStreetMap

Features:

Population trends
Crime statistics
Urban density

Output:

NeighborhoodRisk
Inflation Agent

Purpose:

Estimate replacement cost changes.

Data Sources:

FRED

Features:

Construction inflation
Labor cost indices

Output:

ReplacementCostRisk
Risk Reasoning Agent

Purpose:

Explain risk.

Inputs:

Outputs from all agents.

Responsibilities:

Explain changes
Identify major contributors
Generate narratives
Recommend mitigation actions

The reasoning layer never fabricates data.

Recommendation Agent

Purpose:

Generate preventive actions.

Examples:

Roof risk:

Schedule inspection
Replace shingles

Weather risk:

Trim trees
Secure outdoor equipment

Water risk:

Install leak sensors

Recommendations should be specific and actionable.

Memory Agent

Purpose:

Track changes over time.

Responsibilities:

Store:

Monthly risk scores
Feature values
Recommendations

Support questions such as:

Why did risk increase?
How has risk evolved?


# Tech Stack 

# Language

Python 3.12+

# Data Processing

pandas
numpy
geopandas
Machine Learning
scikit-learn
xgboost
shap

# Agent Framework

LangGraph

# LLM Framework

LangChain

# Database

PostgreSQL

# Vector Database

Chroma

# API Layer

FastAPI

Frontend

Streamlit

Visualization

plotly

# Project Structure
property_guardian_ai/

app/
agents/
models/
schemas/
services/
tools/
memory/
prompts/
workflows/
api/
dashboard/
tests/
docs/
notebooks/
data/

README.md
CLAUDE.md
Pydantic Schemas

Use Pydantic for all interfaces.

Examples:

WeatherRisk
FloodRisk
ClimateRisk
PropertyProfile
RiskAssessment
Recommendation

Avoid dictionaries whenever possible.

Prefer strongly typed objects.

Coding Standards
Type hints required.
Use dataclasses or Pydantic.
Functions should be small.
Avoid hidden side effects.
Favor composition over inheritance.
Write unit tests.
Use structured logging.
Keep agents independent.
Research Goals

Investigate whether agentic AI can enable:

Continuous underwriting
Dynamic property risk monitoring
Explainable risk assessment
Preventive recommendations
Loss prevention
Success Metrics

Technical:

Precision
Recall
ROC-AUC

Operational:

Explainability
Recommendation quality
Risk trend detection

Business:

Reduced loss exposure
Improved underwriting visibility
Increased prevention opportunities