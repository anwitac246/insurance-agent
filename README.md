# Multi-Agent Car Insurance Claim Processing System

## Overview
This is a production-grade, stateful, and modular multi-agent backend system designed to automate end-to-end car insurance claim processing. It leverages Large Language Models (LLMs), visual recognition, Retrieval-Augmented Generation (RAG), and a synthetic internal database to orchestrate a complex workflow from document ingestion to final decision-making.

---

## Architecture

* **Backend Framework**: FastAPI (Asynchronous background task processing)
* **Agent Orchestration**: LangGraph (StateGraph for routing and cyclic flows)
* **LLM Provider**: Groq (`llama-3.3-70b-versatile` for text, `llama-3.2-11b-vision-preview` for damage analysis)
* **Vector Database**: Pinecone (RAG lookups for policy rules and fraud case studies)
* **Relational/Document Database**: MongoDB (Motor async driver) for Claim State persistence and internal "Ground Truth" data.
* **OCR Tools**: Tesseract / PaddleOCR (for extracting raw text from submitted PDFs/images)

### Agent Roles
1. **Orchestrator**: Controls the workflow using LangGraph. It routes states conditionally (e.g., looping back if documents are missing).
2. **Document Processing Agent**: Uses Pydantic-driven structured outputs to extract data from 8 different required documents.
3. **Policy Verification Agent**: Cross-references extracted data against the internal database.
4. **Fraud Detection Agent**: Analyzes historical claims, vector-database case studies, and internal fraud signals.
5. **Decision Agent**: Synthesizes all data to generate an Approved, Rejected, or Escalate-to-Human decision.

---

## Design & Thought Process

### 1. Strict Document Validation (The "No Hallucination" Rule)
Instead of relying entirely on LLMs to guess if a claim is valid, the **Document Agent** was designed with strict Pydantic schemas representing the required documents (Policy, Claim Form, RC, License, FIR, etc.).
- The LLM is forced via `.with_structured_output()` to output structured JSON.
- We then use **deterministic Python logic** to cross-validate the parsed data (e.g., explicit `if policy.vehicle_number != rc.vehicle_number` checks). This ensures high reliability and zero LLM hallucinations during the critical validation phase.

### 2. Synthetic Data Strategy & "Ground Truth" Verification
A major limitation of standard AI demos is that they only look at the documents provided by the user. To make this production-grade, we integrated a real **MongoDB database layer**.
- **The Process**: We built a `generate_synthetic_db.py` script using the `Faker` library. It seeds the database with thousands of relational records: Policyholders, Vehicles, Policies, Past Claims, and Fraud Signals.
- **The Benefit**: 
  - When the **Policy Agent** runs, it takes the extracted policy number and queries the DB to see if the policy actually exists, is active, and if the name on the claim matches the actual owner in the database.
  - When the **Fraud Agent** runs, it actively fetches the vehicle's past claims and checks for pre-existing `fraud_signals` (e.g., "vin_cloning_suspected") to generate an accurate risk score. 

### 3. Modular RAG Implementation
Rather than stuffing all policy rules and edge cases into the agent prompts, we created an `ingest_data.py` script that splits markdown files (`data/policy_data` and `data/fraud_data`) using header-aware chunking and embeds them locally using free HuggingFace `sentence-transformers`. This data is pushed to Pinecone, allowing agents to dynamically pull relevant clauses based on the claim context.

---

## How to Run

### 1. Prerequisites
- Python 3.11+
- Tesseract OCR installed on your system.
- A local **MongoDB** instance running on `localhost:27017`.
- Create a `.env` file in the root directory:
  ```env
  GROQ_API_KEY=your_groq_key
  PINECONE_API_KEY=your_pinecone_key
  PINECONE_ENVIRONMENT=us-east-1
  MONGODB_URI=mongodb://localhost:27017
  DATABASE_NAME=insurance_claims
  ```

### 2. Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Initialize the Databases
First, push the Markdown policy/fraud data to your Pinecone vector database:
```bash
python scripts/ingest_data.py
```

Next, seed the local MongoDB instance with the synthetic policyholder, vehicle, and claims data:
```bash
python scripts/generate_synthetic_db.py
```

### 4. Run the API Server
Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```

### 5. Testing the Pipeline
You can test the extraction and validation logic in isolation using the provided test script:
```bash
python scripts/test_validation.py
```
This will run the Document Agent against dummy text to show exactly how it flags missing fields and tracks missing documents.
