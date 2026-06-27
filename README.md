# **`README.md`**

# AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Source](https://img.shields.io/badge/Source-arXiv%20Preprint-B31B1B)](https://arxiv.org/abs/2606.20041)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/ai_economist_agent)
[![Discipline: CS](https://img.shields.io/badge/Discipline-Computer%20Science-00529B)](https://github.com/chirindaopensource/ai_economist_agent)
[![Discipline: AI](https://img.shields.io/badge/Discipline-Artificial%20Intelligence-00529B)](https://github.com/chirindaopensource/ai_economist_agent)
[![Discipline: Macro](https://img.shields.io/badge/Discipline-Macroeconomics-00529B)](https://github.com/chirindaopensource/ai_economist_agent)
[![Discipline: FinEcon](https://img.shields.io/badge/Discipline-Financial%20Econometrics-00529B)](https://github.com/chirindaopensource/ai_economist_agent)
[![Discipline: Graph](https://img.shields.io/badge/Discipline-Graph%20Theory-00529B)](https://github.com/chirindaopensource/ai_economist_agent)
[![Data: Evidence](https://img.shields.io/badge/Data-Synthetic%20Evidence%20Corpus-lightgrey)](https://github.com/chirindaopensource/ai_economist_agent)
[![Data: Temporal](https://img.shields.io/badge/Data-Synthetic%20Temporal%20Facts-lightgrey)](https://github.com/chirindaopensource/ai_economist_agent)
[![Method: RAG](https://img.shields.io/badge/Method-Retrieval--Augmented%20Generation-orange)](https://github.com/chirindaopensource/ai_economist_agent)
[![Method: KG](https://img.shields.io/badge/Method-Knowledge%20Graphs%20(Neo4j)-orange)](https://neo4j.com/)
[![Method: DSGE](https://img.shields.io/badge/Method-DSGE%20Modeling-orange)](https://github.com/chirindaopensource/ai_economist_agent)
[![Method: MS-VAR](https://img.shields.io/badge/Method-Markov--Switching%20VAR-orange)](https://github.com/chirindaopensource/ai_economist_agent)
[![Method: Agents](https://img.shields.io/badge/Method-Agentic%20Orchestration-orange)](https://github.com/chirindaopensource/ai_economist_agent)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/ai_economist_agent)

**Repository:** `https://github.com/chirindaopensource/ai_economist_agent`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_master_ai_economist_pipeline](#key-callable-execute_master_ai_economist_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project is an independent, professional-grade implementation of the ideas, methodologies, and experimental protocols from the paper titled **"AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models"** by the author:
*   **Masahiro Kato**

This repository provides a complete, end-to-end computational framework for replicating the paper's findings. It addresses a critical epistemic failure in the application of Large Language Models (LLMs) to economic analysis: the generation of fluent narratives that lack formal grounding in macroeconomic theory and explicit mathematical models. 

By implementing a strict architectural separation of concerns, this codebase ensures that LLMs are restricted to planning, retrieval, and narrative synthesis, while all quantitative claims are delegated to deterministic, external macroeconometric solvers. The framework utilizes a typed Neo4j Knowledge Graph to enforce this separation, ultimately delivering **Institutional-Grade Automated Economic Intelligence with Regulatory-Compliant Epistemic Traceability**.

## Theoretical Background

The implemented methods bridge advanced natural language processing with rigorous macroeconometrics and graph theory.

**1. Epistemic Traceability & Agentic Orchestration:**
The pipeline moves beyond standard Retrieval-Augmented Generation (RAG) by structurally prohibiting the LLM from generating quantitative data. Instead, the LLM acts as an orchestrator, selecting approved mathematical models from a Knowledge Graph. The final report is generated from a strict information tuple: $(s, E_s, P_s^{\text{pre}}, C_s, r_s, m_s, O_s, P_s^{\text{post}})$, ensuring every numerical claim is traceable to an executed `ModelRun` object.

**2. Knowledge Graph Topology:**
To prevent "model leakage," the framework employs a typed property graph. Textual evidence (`EvidenceSpan`) is stored entirely separately from mathematical definitions (`ModelSpecification`). This topological isolation guarantees that a RAG-only agent cannot accidentally hallucinate a model execution merely by reading a model's name in a retrieved text document.

**3. Macroeconometrics (DSGE-lite):**
For Application 1 (U.S. Inflation Persistence), the computational layer executes a linearized 3-equation New Keynesian framework:
*   **IS Curve:** $y_t = E_t y_{t+1} - \frac{1}{\sigma}(i_t - E_t \pi_{t+1}) + g_t$
*   **Phillips Curve:** $\pi_t = \beta E_t \pi_{t+1} + \kappa y_t + u_t$
*   **Taylor Rule:** $i_t = \rho_i i_{t-1} + (1-\rho_i)(\phi_\pi \pi_t + \phi_y y_t) + v_t$

The system is solved using Blanchard-Kahn conditions via generalized Schur decomposition, verifying saddle-path stability before simulating a 12-step horizon with AR(1) shocks.

**4. Financial Econometrics & Stress Testing (MS-VAR):**
For Application 2 (CRE Refinancing Stress), the layer executes a 2-state Markov-Switching Vector Autoregression ($Y_t = c_{S_t} + A_{1,S_t} Y_{t-1} + \epsilon_{S_t,t}$). The Hamilton filter computes conditional regime probabilities. The resulting macro-financial paths are deterministically mapped to Basel III/CCAR bank stress metrics, including logistic Probability of Default ($PD_t = \Lambda(\alpha_0 + \alpha_1 p_t^{CRE} + \alpha_2 s_t^{credit})$) and the Common Equity Tier 1 ratio ($CET1_t$).

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/ai_economist_agent/blob/main/ai_economist_agent_ipo_main_9.png" alt="Pipeline Architecture" width="100%">
</div>

## Features

-   **End-to-End Automation:** A singular orchestrator function (`execute_master_ai_economist_pipeline`) executes the entire research protocol from raw data ingestion and schema validation to cryptographic packaging and LaTeX report generation.
-   **Strict Determinism:** Enforces `temperature = 0` and `seed = 42` for the local `llama3.1:8b` inference engine, coupled with deterministic PRNG initialization for all Monte Carlo shock simulations.
-   **Atomic Graph Mutations:** Executes Neo4j database transactions to ensure the "Graph Reset" completely purges prior states, and that "Graph Write-Back" creates `ModelRun` and `ModelOutputPoint` nodes as an all-or-nothing transaction.
-   **Fail-Closed Data Governance:** Implements rigorous pandas schema validation, enforcing extension dtypes (`Int64`, `string`) and executing set-difference operations to guarantee primary/foreign key referential integrity before any LLM inference occurs.
-   **Regulatory-Compliant Provenance:** Serializes all intermediate artifacts and computes SHA-256 cryptographic hashes of the executed computational runner modules, anchoring numeric outputs to the exact Python code executed.

## Methodology Implemented

1.  **Infrastructure Provisioning:** Validates the deterministic LLM endpoint, audits network isolation (enforcing the "no external data" constraint), and provisions the Neo4j database with uniqueness constraints.
2.  **Data Validation & Cleansing:** Validates 12 raw DataFrames against declarative schemas, strips identifier whitespace, normalizes evidence tags, and enforces a strict UTC timezone policy.
3.  **Graph Reset & Loading:** Atomically purges the Knowledge Graph and loads the cleansed data using a strict two-phase strategy (nodes, then edges) to maintain referential integrity.
4.  **Agentic Planning & Retrieval:** The Planner LLM structures the scenario objective. The system executes parallel Text Retrieval (50/50 weighted tag-overlap and confidence) and Pre-Execution Graph Retrieval (constrained single-hop Cypher queries).
5.  **Model Selection & Execution:** The Model-Selection LLM maps the graph context to an execution request. The Computational Layer solves the DSGE/MS-VAR models and simulates the 12-step paths.
6.  **Graph Write-Back & Post-Retrieval:** The execution artifacts are written atomically to the graph. A read-committed query retrieves the full traceability path ($P_s^{\text{post}}$).
7.  **Report Generation & Judging:** Reports are generated under three isolated conditions (LLM-only, RAG-only, Model-grounded). The Judge LLM scores the reports, enforcing a strict logic gate: model grounding requires an executed model object.
8.  **Archival:** All artifacts, runtime environments, and assumption registries are packaged into an immutable provenance directory.

## Core Components (Notebook Structure)

*Note: All orchestrator callables and their constituent helper functions are contained within a singular, comprehensive Jupyter Notebook (`ai_economist_agent_draft.ipynb`).*

The notebook is structured as a logical Directed Acyclic Graph (DAG) of 29 distinct tasks:
1.  Configuration Validation & Infrastructure Provisioning (Tasks 1-3, 11-12)
2.  DataFrame Schema Validation & Referential Integrity (Tasks 4-5)
3.  Row-Level Invariants, Cleansing, & Datetime Normalization (Tasks 6-8)
4.  JSON Payload Validation & Synthetic Data Scaling (Tasks 9-10)
5.  Graph Reset & Pre-Run Materialization (Tasks 13-14)
6.  Text Retrieval & Pre-Execution Graph Retrieval (Tasks 15-16)
7.  Planner & Model-Selection Agent Orchestration (Tasks 17-18)
8.  Computational Runners: DSGE-lite & MS-VAR (Tasks 19-20)
9.  Graph Write-Back & Post-Execution Retrieval (Tasks 21-22)
10. Report Generation & Rubric-Based Judging (Tasks 23-24)
11. End-to-End Orchestration & Application Execution (Tasks 25-27)
12. Provenance Archival & Final Validation Reporting (Tasks 28-29)

## Key Callable: `execute_master_ai_economist_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_master_ai_economist_pipeline`:** This apex orchestrator function coordinates the entire AI Economist Agent experimental protocol. A single call to this function ingests the raw DataFrames, provisions the infrastructure, executes the data cleansing and graph loading, runs the agentic workflow for both Application 1 and Application 2, and packages the results into an immutable provenance archive. It guarantees complete state isolation and prevents look-ahead bias by threading a global assumption registry.

## Prerequisites

-   Python 3.10+
-   Neo4j Database 5.x (Local or AuraDB)
-   Local LLM Inference Server (e.g., Ollama running `llama3.1:8b` on port 11434)
-   Core Python dependencies: `numpy`, `pandas`, `scipy`, `neo4j`, `requests`, `psutil`, `pyyaml`, `faker` (for synthetic testing).

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/ai_economist_agent.git
    cd ai_economist_agent
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires 12 primary raw data structures (pandas DataFrames), adhering strictly to the schemas defined in Table 1 and Table 3 of the methodology:
1.  **`raw_source_documents_df`**: Canonical provenance table for textual sources.
2.  **`raw_evidence_spans_df`**: Primary textual evidence payload (`span_text`, `tags`, `synthetic_confidence`).
3.  **`raw_temporal_facts_df`**: Time-bounding evidence spans (`validity_start`, `validity_end`).
4.  **`raw_shocks_df`**: Encodes scenario shock drivers.
5.  **`raw_variable_concepts_df`**: Canonical variable dictionary.
6.  **`raw_models_df`**: Abstract model families.
7.  **`raw_model_specifications_df`**: The approved, governed unit the LLM is allowed to select (contains `calibration_params_json`).
8.  **`raw_model_equations_df`**: Stores approved LaTeX equations for auditability.
9.  **`raw_model_assumptions_df`**: Stores modeling assumptions distinctly from equations.
10. **`raw_model_documents_df`**: Governance/provenance documents for model approval.
11. **`raw_implementation_functions_df`**: The mapping from approved spec to executable Python function family.
12. **`raw_model_variable_relationships_df`**: Explicit join table enabling safe model selection.

Additionally, the pipeline requires a **`config.yaml`** file, which serves as the deterministic source of truth for all hyperparameters, LLM settings, and mathematical calibrations.

## Usage

Here is the granular, step-by-step guide to executing the end-to-end pipeline for **"AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models"**. This example demonstrates how to synthetically generate the required Knowledge Graph input data, load the study configuration from a YAML file, and execute the full research pipeline using the `execute_master_ai_economist_pipeline` orchestrator.

*Note: It is assumed that all the callables that are defined in the `ai_economist_agent_draft.ipynb` notebook are present in a single Jupyter notebook or execution environment, and that there is no separate folder with ".py" functions. It is also assumed that a valid `config.yaml` file containing the `study_config` dictionary is saved in the current working directory.*

### **Step 1: Synthetic Data Generation (`dataframes_dict`)**

The first requirement is a high-fidelity synthetic representation of the 12 raw input DataFrames. These datasets must adhere strictly to the schemas defined in the study, utilizing pandas extension types (e.g., `string`, `Int64`, `boolean`) to ensure perfect compatibility with the rigorous validation layer (Task 4).

**Methodology:**
1.  **Schema Enforcement:** We explicitly cast columns to their required dtypes (`string`, `datetime64[ns]`, `float64`, `Int64`, `boolean`) to pass the strict `validate_dataframe_schema` checks.
2.  **Referential Integrity:** We ensure that foreign keys (e.g., `doc_id`, `spec_id`, `scenario_id`) perfectly match their corresponding primary keys to pass the `validate_referential_integrity` checks (Task 5).
3.  **Data Assembly:** We construct a master dictionary `dataframes_dict` containing all 12 DataFrames, ready for ingestion by the pipeline.

```python
import pandas as pd
import numpy as np
from typing import Dict, Any
from unittest.mock import MagicMock

# Set deterministic seed for reproducibility
np.random.seed(42)

def generate_synthetic_kg_inputs() -> Dict[str, pd.DataFrame]:
    """
    Generates the 12 synthetic DataFrames required for the AI Economist Agent pipeline.

    Purpose:
        To create a high-fidelity mock dataset that mimics the schema and relational
        properties of the Knowledge Graph inputs defined in Table 1 and Table 3 of
        the manuscript. This allows for the execution of the research pipeline.

    Inputs:
        None.

    Processes:
        1.  Constructs raw dictionaries for each of the 12 required tables.
        2.  Converts dictionaries to pandas DataFrames.
        3.  Explicitly casts columns to strict pandas extension types (string, Int64, boolean)
            and datetime64[ns] to satisfy the Task 4 schema validation.

    Outputs:
        Dict[str, pd.DataFrame]: A dictionary mapping the 12 table names to their
                                 respective DataFrames.

    Raises:
        ValueError: If DataFrame construction fails.
    """
    # 1. Raw Source Documents
    # Canonical provenance table for textual sources.
    raw_source_documents_df = pd.DataFrame({
        "doc_id": pd.Series(["DOC_MACRO_REV_2024", "DOC_FED_OUTLOOK_2024"], dtype="string"),
        "title": pd.Series(["Macroeconomic Review and Stress Testing Analysis", "Federal Reserve Monetary Policy Outlook"], dtype="string"),
        "publication_date": pd.to_datetime(["2024-05-15", "2024-06-01"]),
        "document_type": pd.Series(["internal_report", "external_outlook"], dtype="string"),
        "source_institution": pd.Series(["Internal Economics Division", "Federal Reserve Board"], dtype="string")
    })

    # 2. Raw Evidence Spans
    # Primary textual evidence payload used by retrieval.
    raw_evidence_spans_df = pd.DataFrame({
        "evidence_id": pd.Series(["EV_1571", "EV_841", "EV_1161", "EV_1321", "EV_1801", "EV_721", "EV_1967", "EV_2028", "EV_2107", "EV_1197"], dtype="string"),
        "doc_id": pd.Series(["DOC_MACRO_REV_2024", "DOC_MACRO_REV_2024", "DOC_MACRO_REV_2024", "DOC_MACRO_REV_2024", "DOC_FED_OUTLOOK_2024", "DOC_FED_OUTLOOK_2024", "DOC_MACRO_REV_2024", "DOC_MACRO_REV_2024", "DOC_FED_OUTLOOK_2024", "DOC_MACRO_REV_2024"], dtype="string"),
        "span_text": pd.Series([
            "A more restrictive policy path usually lifts nominal yields, although the term premium may vary by regime.",
            "Global oil supply shocks tend to raise headline inflation before the pass-through fades.",
            "Credit losses reduce common equity capital and lower the stressed CET1 ratio.",
            "Declines in CRE values raise loss-given-default and can increase credit losses.",
            "China growth surprises transmit to global trade volumes through supply-chain and commodity channels.",
            "Funding stress and confidence effects can increase uninsured deposit outflows.",
            "Funding stress and confidence effects can increase uninsured deposit outflows.",
            "Deposit outflows increase net cash outflows and reduce liquidity coverage ratios.",
            "Funding stress and confidence effects can increase uninsured deposit outflows.",
            "Funding stress and confidence effects can increase uninsured deposit outflows."
        ], dtype="string"),
        "tags": pd.Series([
            "monetary_policy,yields,term_premium", "inflation,supply_shock,oil", "credit_loss,bank_capital,cet1",
            "cre,lgd,credit_loss", "global_growth,trade,commodities", "funding_stress,deposits,liquidity",
            "funding_stress,deposits,liquidity", "deposits,liquidity,lcr", "funding_stress,deposits,liquidity",
            "funding_stress,deposits,liquidity"
        ], dtype="string"),
        "synthetic_confidence": pd.Series([0.935, 0.933, 0.940, 0.939, 0.936, 0.932, 0.939, 0.938, 0.937, 0.931], dtype="float64"),
        "char_start_position": pd.Series([1024, 2048, 3072, 4096, 512, 768, 1280, 1536, 640, 896], dtype="Int64"),
        "char_end_position": pd.Series([1132, 2144, 3152, 4184, 620, 840, 1352, 1620, 712, 968], dtype="Int64")
    })

    # 3. Raw Temporal Facts
    # Time-bounding evidence spans: observation time vs validity window.
    raw_temporal_facts_df = pd.DataFrame({
        "fact_id": pd.Series(["TF_1571_01", "TF_841_01", "TF_1161_01", "TF_1321_01", "TF_1801_01", "TF_721_01", "TF_1967_01", "TF_2028_01", "TF_2107_01", "TF_1197_01"], dtype="string"),
        "evidence_id": pd.Series(["EV_1571", "EV_841", "EV_1161", "EV_1321", "EV_1801", "EV_721", "EV_1967", "EV_2028", "EV_2107", "EV_1197"], dtype="string"),
        "observation_time": pd.to_datetime(["2024-04-30T00:00:00Z", "2024-03-31T00:00:00Z", "2024-04-15T00:00:00Z", "2024-04-20T00:00:00Z", "2024-05-10T00:00:00Z", "2024-05-05T00:00:00Z", "2024-04-25T00:00:00Z", "2024-04-28T00:00:00Z", "2024-05-02T00:00:00Z", "2024-04-18T00:00:00Z"], utc=True),
        "validity_start": pd.to_datetime(["2024-05-01T00:00:00Z", "2024-04-01T00:00:00Z", "2024-04-15T00:00:00Z", "2024-04-20T00:00:00Z", "2024-05-10T00:00:00Z", "2024-05-05T00:00:00Z", "2024-04-25T00:00:00Z", "2024-04-28T00:00:00Z", "2024-05-02T00:00:00Z", "2024-04-18T00:00:00Z"], utc=True),
        "validity_end": pd.to_datetime(["2025-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2025-06-30T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z", "2024-12-31T00:00:00Z"], utc=True),
        "temporal_description": pd.Series(["Yield response to policy path observed in Q2 2024 regime", "Oil supply shock pass-through window", "CET1 capital impact observation", "CRE valuation decline period", "China growth spillover transmission window", "Funding stress confidence channel activation", "Deposit outflow stress episode", "Liquidity coverage stress measurement", "Uninsured deposit flight window", "Bank funding pressure period"], dtype="string")
    })

    # 4. Raw Shocks
    # Encodes scenario shock drivers.
    raw_shocks_df = pd.DataFrame({
        "shock_id": pd.Series(["SHOCK_MONETARY_TIGHT", "SHOCK_COST_PUSH"], dtype="string"),
        "scenario_id": pd.Series(["SCEN_APP1_INFLATION", "SCEN_APP1_INFLATION"], dtype="string"),
        "shock_description": pd.Series(["Restrictive monetary policy path lifting nominal yields", "Persistent services inflation cost-push shock"], dtype="string"),
        "shock_type": pd.Series(["monetary_policy", "supply_side"], dtype="string"),
        "severity_level": pd.Series([3, 2], dtype="Int64")
    })

    # 5. Raw Variable Concepts
    # Canonical variable dictionary.
    raw_variable_concepts_df = pd.DataFrame({
        "concept_id": pd.Series(["VAR_INFLATION", "VAR_FFR", "VAR_10Y_YIELD", "VAR_OUTPUT_GAP"], dtype="string"),
        "scenario_id": pd.Series(["SCEN_APP1_INFLATION", "SCEN_APP1_INFLATION", "SCEN_APP1_INFLATION", "SCEN_APP1_INFLATION"], dtype="string"),
        "variable_name": pd.Series(["U.S. headline inflation", "Federal funds rate", "U.S. ten-year yield", "U.S. output gap"], dtype="string"),
        "variable_symbol": pd.Series(["π_t", "i_t", "i_t^(10)", "y_t"], dtype="string"),
        "measurement_unit": pd.Series(["percent_annual", "percent_annual", "percent_annual", "percent_deviation"], dtype="string"),
        "is_target_variable": pd.Series([True, True, True, True], dtype="boolean")
    })

    # 6. Raw Models
    # Abstract model families.
    raw_models_df = pd.DataFrame({
        "model_id": pd.Series(["MOD_DSGE_LITE", "MOD_VAR", "MOD_REGIME_SWITCH"], dtype="string"),
        "model_name": pd.Series(["Dynamic Stochastic General Equilibrium (Lite)", "Vector Autoregression", "Markov-Switching Vector Autoregression"], dtype="string"),
        "model_category": pd.Series(["structural", "reduced_form", "nonlinear"], dtype="string"),
        "complexity_level": pd.Series([3, 1, 4], dtype="Int64")
    })

    # 7. Raw Model Specifications
    # The approved, governed unit the LLM is allowed to select.
    raw_model_specifications_df = pd.DataFrame({
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_VAR_001", "MSPEC_REGIME_SWITCHING_005"], dtype="string"),
        "model_id": pd.Series(["MOD_DSGE_LITE", "MOD_VAR", "MOD_REGIME_SWITCH"], dtype="string"),
        "spec_description": pd.Series(["3-equation New Keynesian model with AR(1) shocks and term structure.", "Reduced-form Vector Autoregression with 2 lags.", "2-state Markov-Switching VAR for normal and stress dynamics."], dtype="string"),
        "calibration_params_json": pd.Series(['{"sigma": 1.5, "beta": 0.99, "kappa": 0.15, "phi_pi": 1.5, "phi_y": 0.125, "rho_i": 0.8, "rho_g": 0.9, "rho_u": 0.85, "rho_v": 0.7, "rho_tau": 0.6}', '{"lag_order": 2}', '{"p11": 0.95, "p22": 0.85, "n_regimes": 2}'], dtype="string")
    })

    # 8. Raw Model Equations
    # Stores approved equations for auditability and governance.
    raw_model_equations_df = pd.DataFrame({
        "eq_id": pd.Series(["EQ_IS_CURVE", "EQ_NK_PHILLIPS", "EQ_TAYLOR_RULE", "EQ_TERM_STRUCTURE", "EQ_MSVAR_STATE"], dtype="string"),
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_REGIME_SWITCHING_005"], dtype="string"),
        "equation_name": pd.Series(["IS Curve", "New Keynesian Phillips Curve", "Monetary Policy Rule (Taylor Rule)", "Term Structure Equation", "MS-VAR State Equation"], dtype="string"),
        "latex_equation": pd.Series([r"y_t = E_t y_{t+1} - \frac{1}{\sigma}(i_t - E_t \pi_{t+1}) + g_t", r"\pi_t = \beta E_t \pi_{t+1} + \kappa y_t + u_t", r"i_t = \rho_i i_{t-1} + (1-\rho_i)(\phi_\pi \pi_t + \phi_y y_t) + v_t", r"i_t^{(10)} = \frac{1}{10} \sum_{j=0}^{9} E_t i_{t+j} + \tau_t", r"Y_t = c_{S_t} + A_{1,S_t} Y_{t-1} + \epsilon_{S_t,t}"], dtype="string"),
        "equation_order": pd.Series([1, 2, 3, 4, 1], dtype="Int64")
    })

    # 9. Raw Model Assumptions
    # Stores modeling assumptions distinctly from equations.
    raw_model_assumptions_df = pd.DataFrame({
        "assumption_id": pd.Series(["ASSUM_LINEAR", "ASSUM_AR1_SHOCKS", "ASSUM_MARKOV_FIRST_ORDER"], dtype="string"),
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_REGIME_SWITCHING_005"], dtype="string"),
        "assumption_text": pd.Series(["Model is linearized around a zero steady state.", "Exogenous shocks follow AR(1) processes.", "Regime transitions follow a first-order Markov process."], dtype="string"),
        "assumption_type": pd.Series(["functional_form", "stochastic_process", "stochastic_process"], dtype="string")
    })

    # 10. Raw Model Documents
    # Governance / provenance documents for model approval.
    raw_model_documents_df = pd.DataFrame({
        "model_document_id": pd.Series(["MDOC_DSGE_001", "MDOC_MSVAR_001"], dtype="string"),
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_REGIME_SWITCHING_005"], dtype="string"),
        "title": pd.Series(["DSGE Model Risk Validation Report", "MS-VAR Stress Testing Methodology"], dtype="string"),
        "document_text": pd.Series(["Approved for inflation scenario analysis.", "Approved for CRE stress testing."], dtype="string"),
        "version": pd.Series(["v1.2", "v2.0"], dtype="string")
    })

    # 11. Raw Implementation Functions
    # The mapping from approved spec -> executable function family.
    raw_implementation_functions_df = pd.DataFrame({
        "impl_id": pd.Series(["IMPL_DSGE_RUNNER", "IMPL_MSVAR_RUNNER"], dtype="string"),
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_REGIME_SWITCHING_005"], dtype="string"),
        "execution_tool_name": pd.Series(["dsge_lite_runner", "regime_switching_runner"], dtype="string"),
        "solver_method": pd.Series(["Blanchard-Kahn", "Hamilton Filter"], dtype="string"),
        "horizon_steps": pd.Series([12, 12], dtype="Int64"),
        "python_module_path": pd.Series(["computational_layer.dsge_lite", "computational_layer.regime_switching"], dtype="string")
    })

    # 12. Raw Model Variable Relationships
    # Explicit join table that makes model selection safe and queryable.
    raw_model_variable_relationships_df = pd.DataFrame({
        "relationship_id": pd.Series(["REL_0001", "REL_0002", "REL_0003", "REL_0004"], dtype="string"),
        "spec_id": pd.Series(["MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004", "MSPEC_DSGE_LITE_004"], dtype="string"),
        "concept_id": pd.Series(["VAR_INFLATION", "VAR_FFR", "VAR_10Y_YIELD", "VAR_OUTPUT_GAP"], dtype="string"),
        "relationship_type": pd.Series(["OUTPUTS_VARIABLE", "OUTPUTS_VARIABLE", "OUTPUTS_VARIABLE", "OUTPUTS_VARIABLE"], dtype="string"),
        "is_endogenous": pd.Series([True, True, True, True], dtype="boolean"),
        "variable_role": pd.Series(["forward_looking", "forward_looking", "expectation", "forward_looking"], dtype="string")
    })

    # Assemble the master dictionary
    dataframes_dict = {
        "raw_source_documents_df": raw_source_documents_df,
        "raw_evidence_spans_df": raw_evidence_spans_df,
        "raw_temporal_facts_df": raw_temporal_facts_df,
        "raw_shocks_df": raw_shocks_df,
        "raw_variable_concepts_df": raw_variable_concepts_df,
        "raw_models_df": raw_models_df,
        "raw_model_specifications_df": raw_model_specifications_df,
        "raw_model_equations_df": raw_model_equations_df,
        "raw_model_assumptions_df": raw_model_assumptions_df,
        "raw_model_documents_df": raw_model_documents_df,
        "raw_implementation_functions_df": raw_implementation_functions_df,
        "raw_model_variable_relationships_df": raw_model_variable_relationships_df
    }

    return dataframes_dict

# Generate the synthetic dataset dictionary
dataframes_dict = generate_synthetic_kg_inputs()

# Preview the data to verify schema compliance
print("Synthetic Knowledge Graph Inputs Generated Successfully.")
print(f"Total DataFrames created: {len(dataframes_dict)}")
print("\nPreview of raw_evidence_spans_df:")
print(dataframes_dict["raw_evidence_spans_df"].head(2))
```

### **Step 2: Loading the Configuration (`config.yaml`)**

The study relies on a deterministic configuration file (`config.yaml`) that defines all hyperparameters, LLM settings, and mathematical calibrations. We assume this file exists in the working directory and contains the `study_config` dictionary defined in the prompt.

**Methodology:**
1.  **File I/O:** Open `config.yaml` in read mode.
2.  **Parsing:** Use `yaml.safe_load` to convert the YAML structure into a nested Python dictionary.
3.  **Verification:** Print keys to ensure the file was loaded correctly.

```python
import yaml

def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the study configuration parameters from a YAML file into a Python dictionary.

    Purpose:
        To ingest the deterministic hyperparameters, LLM settings, and mathematical
        calibrations defined in the external configuration file. This ensures
        reproducibility by separating code from configuration.

    Inputs:
        filepath (str): The relative or absolute path to the YAML configuration file.
                        Default is "config.yaml".

    Processes:
        1.  File Access: Attempts to open the specified file in read mode.
        2.  Parsing: Uses PyYAML's safe_load to parse the YAML structure securely.
        3.  Validation: Catches and handles FileNotFoundError and YAMLError.

    Outputs:
        Dict[str, Any]: A nested dictionary containing the study configuration.

    Raises:
        TypeError: If filepath is not a string.
        FileNotFoundError: If the config.yaml file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    # Validate input type to ensure string path
    if not isinstance(filepath, str):
        # Raise TypeError if the path is not a string
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}.")

    try:
        # Open the file stream in read mode with UTF-8 encoding
        with open(filepath, "r", encoding="utf-8") as file:
            # Parse the YAML content safely into a Python dictionary
            config = yaml.safe_load(file)

        # Log success to the console for user visibility
        print(f"Successfully loaded configuration from {filepath}")

        # Return the parsed configuration dictionary
        return config

    except FileNotFoundError:
        # Raise an explicit error if the configuration file is missing
        raise FileNotFoundError(f"Critical Error: {filepath} not found in the working directory.")
    except yaml.YAMLError as e:
        # Handle YAML syntax errors explicitly
        print(f"Error parsing YAML file {filepath}: {e}")
        # Re-raise the exception to halt execution on corrupted config
        raise

# Load the configuration from the working directory into the 'config' dictionary
# Note: Ensure 'config.yaml' is in your working directory with the content provided previously.
# For the purpose of this self-contained example, if the file is missing, we will use the
# study_config dictionary defined in the prompt.
try:
    config = load_study_configuration("config.yaml")
except FileNotFoundError:
    print("Falling back to in-memory study_config for demonstration purposes.")
    # Assuming study_config is defined in the global namespace from the prompt
    config = study_config 
```

### **Step 3: Executing the Pipeline (`execute_master_ai_economist_pipeline`)**

With the data (`dataframes_dict`) and configuration (`config`) in memory, we invoke the top-level orchestrator. This function manages the entire lifecycle: infrastructure validation, data cleansing, synthetic scaling, application execution (DSGE-lite and MS-VAR), and final archival.

*Note: For this example to run without crashing in a standard Python environment lacking a live Neo4j database and a local Llama 3.1 endpoint, we instantiate mock objects for the `neo4j_driver` and `llm_client`. In a production environment, these would be live connection objects.*

**Methodology:**
1.  **Mocking Infrastructure:** Instantiate `MagicMock` objects for the database driver and LLM client to allow the pipeline to execute its logic without external dependencies.
2.  **Function Call:** Pass the DataFrames, configuration, and mock objects to `execute_master_ai_economist_pipeline`.
3.  **Output Handling:** The function returns a master dictionary containing all intermediate logs, application results, and the final validation report.

```python
# ==============================================================================
# Execution of the End-to-End AI Economist Agent Pipeline
# ==============================================================================

if __name__ == "__main__":
    # Ensure we have valid inputs before initiating the heavy computational pipeline
    if dataframes_dict and config:
        
        # Log the initiation of the pipeline
        print("\nInitiating AI Economist Agent Study Pipeline...")
        
        # Instantiate mock objects for the external infrastructure dependencies
        # This allows the example to run without a live Neo4j database or LLM endpoint
        mock_neo4j_driver = MagicMock()
        
        try:
            # Execute the master pipeline orchestrator
            # This will trigger the full sequence of Tasks 1 through 29
            # Note: execute_master_ai_economist_pipeline is assumed to be in the namespace
            master_artifacts = execute_master_ai_economist_pipeline(
                dataframes_dict=dataframes_dict,
                study_config=config,
                neo4j_driver=mock_neo4j_driver,
                base_archive_path="./provenance_archives"
            )
            
            # ==============================================================================
            # Inspecting the Outputs
            # ==============================================================================
            
            # Print a visual separator for the completion block
            print("\n" + "="*80)
            # Announce successful completion
            print("STUDY EXECUTION COMPLETE")
            # Print a visual separator
            print("="*80)
            
            # 1. Accessing Data Preparation Logs
            print("\n[Data Preparation Phase]")
            if "data_preparation_logs" in master_artifacts:
                logs = master_artifacts["data_preparation_logs"]
                print(f"Identifier Audit Log Entries: {len(logs.get('identifier_audit_log', []))}")
                print(f"Tag Lineage Log Entries: {len(logs.get('tag_lineage_log', []))}")
                
            # 2. Accessing Application 1 Results (DSGE-lite)
            print("\n[Application 1: U.S. Inflation Persistence]")
            if "application_1_results" in master_artifacts:
                app1_res = master_artifacts["application_1_results"]
                print("Model Selected:", app1_res.get("execution_request", {}).get("model_specification"))
                print("Judge Scores (Model-Grounded):", app1_res.get("scores_model_grounded"))
                
            # 3. Accessing Application 2 Results (MS-VAR)
            print("\n[Application 2: U.S. CRE Refinancing Stress]")
            if "application_2_results" in master_artifacts:
                app2_res = master_artifacts["application_2_results"]
                print("Model Selected:", app2_res.get("execution_request", {}).get("model_specification"))
                print("Judge Scores (Model-Grounded):", app2_res.get("scores_model_grounded"))

            # 4. Printing the Final Validation Report
            print("\n" + "="*80)
            print("FINAL REPRODUCTION VALIDATION REPORT")
            print("="*80)
            print(master_artifacts.get("final_validation_report", "Report not generated."))
            
        except Exception as e:
            # Catch and display any fatal errors that occurred during pipeline execution
            print(f"\nPipeline Execution Failed: {str(e)}")
            
    else:
        # Handle the case where data generation or config loading failed
        print("Error: Missing synthetic data or configuration. Cannot proceed.")
```

## Output Structure

The pipeline returns a comprehensive, cryptographically hashed `.tar.gz` archive containing the following key artifacts:
-   **`01_scenario.txt`**: The raw narrative text of the economic scenario.
-   **`02_planner_output.json`**: The structured analysis plan generated by the Planner Agent.
-   **`03_retrieved_evidence.json`**: The top-6 retrieved evidence spans with their synthetic confidence scores.
-   **`04_pre_run_paths.txt`**: The pre-execution graph paths retrieved from Neo4j.
-   **`05_model_context.json`**: The executable model context provided to the Model-Selection Agent.
-   **`06_execution_request.json`**: The structured execution request ($r_s$) generated by the Model-Selection Agent.
-   **`07_model_run_metadata.json`**: The metadata for the executed `ModelRun` node, including the generated `run_id`.
-   **`08_model_output_table.csv`**: The 12-step numerical paths for the macro-financial variables.
-   **`08_bank_metrics.json`**: The computed Basel III/CCAR bank stress metrics (Application 2 only).
-   **`09_post_run_paths.txt`**: The post-execution graph paths retrieved from Neo4j, summarizing extremum values.
-   **`10_report_*.txt`**: The generated economic reports for the three experimental conditions (LLM-only, RAG-only, Model-grounded).
-   **`11_scores_*.json`**: The rubric scores generated by the Judge Agent for the three experimental conditions.
-   **`runtime_environment.json`**: The runtime environment, library versions, and SHA-256 cryptographic hashes of the executed computational runner modules.
-   **`implementation_assumptions.json`**: The assumption registry documenting all manuscript-unspecified choices (e.g., diagonal MS-VAR matrices, retrieval weights).
-   **`reproduction_validation_report.txt`**: The final validation report summarizing the reproduction outcomes and numerical limitations.

## Project Structure

```
ai_economist_agent/
│
├── ai_economist_agent_draft.ipynb              # Main implementation notebook containing all callables
├── config.yaml                                 # Master configuration file (Budgets, Hyperparameters, Seeds)
├── requirements.txt                            # Python package dependencies
│
├── provenance_archives/                        # Auto-generated archival directory
│   ├── APP_1_RUN_SCEN_APP1_INFLATION_20260618T120000Z/
│   │   ├── 01_scenario.txt
│   │   ├── 02_planner_output.json
│   │   ├── 03_retrieved_evidence.json
│   │   ├── 04_pre_run_paths.txt
│   │   ├── 05_model_context.json
│   │   ├── 06_execution_request.json
│   │   ├── 07_model_run_metadata.json
│   │   ├── 08_model_output_table.csv
│   │   ├── 08_model_output_table.md
│   │   ├── 09_post_run_paths.txt
│   │   ├── 10_report_llm_only.txt
│   │   ├── 10_report_rag_only.txt
│   │   ├── 10_report_model_grounded.txt
│   │   ├── 11_scores_llm_only.json
│   │   ├── 11_scores_rag_only.json
│   │   ├── 11_scores_model_grounded.json
│   │   ├── runtime_environment.json
│   │   ├── implementation_assumptions.json
│   │   └── reproduction_validation_report.txt
│   │
│   └── APP_2_RUN_SCEN_APP2_CRE_20260618T120500Z/
│       ├── ... (similar structure to APP_1)
│       └── 08_bank_metrics.json
│
├── LICENSE                                     # MIT Project License File
└── README.md                                   # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Researchers and engineers can modify study parameters such as:
-   **LLM Inference Settings:** Adjust the `model_name`, `temperature`, `seed`, and `max_tokens` in the `llm_execution_parameters` block to experiment with different local models or sampling strategies.
-   **Retrieval Parameters:** Modify `text_evidence_top_k`, `model_context_top_k`, `tag_overlap_weight`, and `confidence_score_weight` in the `retrieval_graph_parameters` block to tune the RAG retrieval logic.
-   **Graph Database Policies:** Toggle `clear_graph_on_init`, `allow_neighborhood_expansion`, and `require_task_specific_queries` in the `retrieval_graph_parameters` block to experiment with different graph traversal constraints.
-   **Computational Solver Settings:** Adjust `model_horizon_steps`, `random_seed`, `bk_tolerance`, and `hamilton_filter_tolerance` in the `computational_solver_parameters` block to fine-tune the mathematical execution layer.
-   **Mathematical Calibrations:** Modify the parameters in the `dsge_lite_parameters`, `regime_switching_parameters`, and `bank_stress_parameters` blocks to simulate different economic scenarios or stress testing methodologies.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. 

**Strict Engineering Standards:** Adherence to PEP-8, strict type hinting (`typing` module), and the draconian requirement for line-by-line in-text comments that explain the mathematical or logical purpose of every single line of code is strictly required for all pull requests to maintain the implementation-grade standard of this repository.

## Recommended Extensions

Future extensions, building upon this foundational framework, could include:
-   **Real-Time Data Integration:** Integrating real-time economic data feeds (e.g., FRED, Bloomberg) to dynamically update the `TemporalFact` nodes and drive automated nowcasting reports.
-   **Expanded Model Catalog:** Expanding the computational layer to include Heterogeneous Agent New Keynesian (HANK) models, Global Vector Autoregressions (GVAR), or deep learning surrogates for nonlinear macro-financial propagation.
-   **Multi-Agent Debate:** Implementing a multi-agent debate mechanism where multiple Report Generator agents synthesize the information tuple and a meta-agent adjudicates the final narrative.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@misc{kato2026aieconomist,
  title={AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models},
  author={Kato, Masahiro},
  howpublished={arXiv preprint arXiv:2606.20041},
  year={2026},
  url={https://arxiv.org/abs/2606.20041}
}
```

For the implementation itself, you may cite this repository:
```bibtex
@misc{chirinda2026aieconomistagent,
  author = {Chirinda, Craig S.},
  title = {AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chirindaopensource/ai_economist_agent}}
}
```

## Acknowledgments

-   Credit to **Masahiro Kato** for the foundational theoretical framework, the rigorous epistemic design, and the synthesis of agentic orchestration with macroeconometric modeling.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **NumPy**, **Pandas**, **SciPy**, and **Neo4j** contributors.

--

*This README was generated based on the structure and content of the `ai_economist_agent_draft.ipynb` notebook and follows best practices for research software documentation.*
