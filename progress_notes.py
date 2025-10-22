# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm
import os

# --- YOU MUST DEFINE full_rawdata HERE ---
full_rawdata = "Progress Note for Mr. Lee, 2024-02-01. Subjective: Patient states less pain. Vitals: Temp 37.2. New Labs: Creatinine 1.2 (stable). Plan: Continue current antibiotics. Disposition: Expected discharge tomorrow."

# Cell 3: Chunking function and Execution
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(full_rawdata)
print(f"Total chunks created: {len(chunks)}")

# Cell 4: Embeddings using Hugging Face (fast) and Execution
model = SentenceTransformer('all-MiniLM-L6-v2')

# Batch encode to avoid memory issues
batch_size = 64
embeddings = []
for i in tqdm(range(0, len(chunks), batch_size)):
    batch_chunks = chunks[i:i+batch_size]
    batch_emb = model.encode(batch_chunks)
    embeddings.append(batch_emb)

embeddings = np.vstack(embeddings)
print(f"Embedding dimension: {embeddings.shape[1]}")

# Cell 5: FAISS index and Execution
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))
print(f"FAISS index contains {index.ntotal} vectors")

# ----------------- USE CASE 4: PROGRESS NOTE SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (Progress Note)
def retrieve_rag_progress(query, top_k=3):
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for Progress Note
def generate_prompt_for_progress():
    """Generates the specific prompt for extracting Progress Note data."""
    retrieved_text = retrieve_rag_progress(
        "Extract subjective, objective, assessment, and plan from today's progress note."
    )
    
    required_structure = """
Progress Note Information:
[Name, MRN, DOB, Age, Date of Service, Service, Author]
Subjective:
[Patient's reported status, overnight events, changes in condition]
Objective:
Vital Signs: [Current vitals with values]
Physical Exam: [Daily examination findings]
Labs: [New laboratory results with values and dates]
Imaging: [New imaging results and interpretations]
I/O: [Intake and output data if documented]
Assessment:
[Daily assessment of each active problem]
Plan:
[Daily management plan for each active problem]
Medication Changes:
[Any medications started, stopped, or dose-adjusted]
Consultant Updates:
[Recent consultant recommendations or findings]
Disposition Planning:
[Any updates to discharge planning or disposition]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided progress note into the exact structure below. 
Use ONLY information explicitly stated in the text. Do not infer, interpret, or
add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Focus on daily changes and updates
- Copy exact medical terminology from source text
- Include all new lab/imaging results
- Document medication changes precisely
- Capture consultant updates verbatim
- Note any changes in patient status
- Maintain problem-oriented approach

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)",
)

prompt = generate_prompt_for_progress()

# Execution Placeholder for demonstration
llm_output = '```json\n{"extracted_clinical_data": "Progress Note Information: Mr. Lee, Date of Service 2024-02-01\\nSubjective: Patient states less pain.\\nObjective: Vital Signs: Temp 37.2. Labs: Creatinine 1.2 (stable).\\nPlan: Continue current antibiotics.\\nDisposition Planning: Expected discharge tomorrow."}\n```'


# Cell 9: Clean LLM output and parse JSON (Progress Note)
def process_llm_output_progress_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted Progress Note Data ---")
        print(extracted_text)
        print("----------------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_progress_json(llm_output)
