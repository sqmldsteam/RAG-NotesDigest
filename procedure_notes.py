# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm
import os

# Cell 3: Chunking function and Execution
def chunk_text(text, chunk_size=800):
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

# ----------------- USE CASE 5: PROCEDURE NOTE SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (Procedure Note)
def retrieve_rag_procedure(query, top_k=3):
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for Procedure Note
def generate_prompt_for_procedure():
    """Generates the specific prompt for extracting Procedure Note data."""
    retrieved_text = retrieve_rag_procedure(
        "Extract procedure details including indication, technique, findings, and post-procedure plan."
    )
    
    required_structure = """
Procedure Information:
[Patient Name, MRN, DOB, Procedure Date, Procedure Type, Performing Physician]
Indication:
[Medical reason for performing the procedure]
Description/Technique:
[Detailed description of how procedure was performed]
Findings:
[Objective results and observations during procedure]
Impression/Diagnosis:
[Interpreting clinician's conclusion or diagnosis]
Complications:
[Any complications encountered]
Post-Procedure Plan:
[Instructions for post-procedure care and monitoring]
Specimens:
[Any specimens sent to pathology]
Medications:
[Medications administered during procedure]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided procedure note into the exact structure below. 
Use ONLY information explicitly stated in the text. Do not infer, interpret, or
add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Copy exact procedural terminology from source text
- Include all technical details of procedure
- Document all findings verbatim
- Preserve clinician's interpretations exactly
- Note any complications precisely
- Capture post-procedure instructions completely
- List all medications administered during procedure

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="https://api.groq.com/openai/v1",
)

prompt = generate_prompt_for_procedure()

# Execution Placeholder for demonstration
llm_output = '```json\n{"extracted_clinical_data": "Procedure Information: P. Rodriguez, Procedure Date 2024-03-10, Procedure Type Lumbar Puncture.\\nIndication: Rule out meningitis.\\nDescription/Technique: Sterile prep, 20G needle, clear fluid obtained.\\nComplications: None.\\nPost-Procedure Plan: Supine for 1 hour."}\n```'


# Cell 9: Clean LLM output and parse JSON (Procedure Note)
def process_llm_output_procedure_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted Procedure Note Data ---")
        print(extracted_text)
        print("-----------------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_procedure_json(llm_output)
