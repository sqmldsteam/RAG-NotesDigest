# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm
import os

# --- YOU MUST DEFINE full_rawdata HERE ---
full_rawdata = "ED Note for S. Chen. Arrival 14:30. Triage: Level 3. CC: Chest Pain. Triage Vitals: HR 95, BP 130/80. ED Course: Given Aspirin. Labs/ECG ordered. Assessment: Atypical Chest Pain. Disposition: Discharged home with follow-up."

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

# ----------------- USE CASE 6: ED NOTE SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (ED Note)
def retrieve_rag_ed(query, top_k=4):
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for ED Note
def generate_prompt_for_ed():
    """Generates the specific prompt for extracting ED Note data."""
    retrieved_text = retrieve_rag_ed(
        "Extract emergency department note details including chief complaint, course, diagnostics, and disposition."
    )
    
    required_structure = """
ED Note Information:
[Patient Name, MRN, DOB, Age, ED Arrival Date/Time, Triage Level, Treating Physician]
Chief Complaint:
[Primary reason for ED visit]
History of Present Illness:
[Detailed narrative of current emergency]
Triage Vitals:
[Initial vital signs in ED]
ED Course:
[Sequence of events in ED including: Initial assessment, Diagnostic workup, Treatments administered, Consultant involvement, Patient response]
ED Physical Exam:
[Focused emergency examination findings]
ED Diagnostic Results:
Labs: [Emergency laboratory findings]
Imaging: [Emergency imaging results]
ECG: [Electrocardiogram findings if performed]
ED Assessment:
[Emergency physician's diagnosis and impressions]
ED Plan:
[Disposition and follow-up instructions]
Disposition:
[Admit/Discharge/Transfer with destination]
ED Medications:
[Medications administered in ED]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided emergency department note into the exact structure below. 
Use ONLY information explicitly stated in the text. Do not infer, interpret, or
add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Focus on emergency presentation and management
- Copy exact emergency medicine terminology
- Include all triage and initial assessment data
- Document emergency interventions precisely
- Capture disposition decision and rationale
- List all ED-administered medications
- Maintain chronological ED course narrative

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)",
)

prompt = generate_prompt_for_ed()

# Execution Placeholder for demonstration
llm_output = '```json\n{"extracted_clinical_data": "ED Note Information: S. Chen, ED Arrival Date/Time 14:30, Triage Level 3.\\nChief Complaint: Chest Pain.\\nTriage Vitals: HR 95, BP 130/80.\\nED Course: Given Aspirin. Labs/ECG ordered.\\nED Assessment: Atypical Chest Pain.\\nDisposition: Discharged home with follow-up."}\n```'


# Cell 9: Clean LLM output and parse JSON (ED Note)
def process_llm_output_ed_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted ED Note Data ---")
        print(extracted_text)
        print("----------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_ed_json(llm_output)
