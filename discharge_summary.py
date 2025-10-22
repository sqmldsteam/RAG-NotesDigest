# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm
import os

# --- YOU MUST DEFINE full_rawdata HERE ---
full_rawdata = "Patient Jane Smith, MRN 654321, discharged on 2023-10-25. Admission Diagnosis: Pneumonia. Discharge Diagnosis: Resolved Pneumonia. Hospital Course: Started on Azithromycin. Follow-up with PCP in 7 days."

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

# ----------------- USE CASE 2: DISCHARGE SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (Discharge Summary)
def retrieve_rag_discharge(query, top_k=5):
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for Discharge Summary
def generate_prompt_for_discharge():
    """Generates the specific prompt for extracting Discharge Summary data."""
    retrieved_text = retrieve_rag_discharge(
        "Extract all components of a discharge summary including course of stay, diagnoses, medications, and follow-up plan."
    )
    
    required_structure = """
Patient Information:
[Name, MRN, DOB, Age, Sex, PCP, Attending, Admit Date, Discharge Date, Location, Discharge Physician]
Admission Diagnoses:
[Bulleted list of ALL admission diagnoses and conditions exactly as documented]
Discharge Diagnoses:
Principal Problem: [Primary discharge diagnosis]
Active Problems: [Comprehensive list of all active problems at discharge]
Hospital Course:
[Detailed chronological narrative including: Reason for admission, Diagnostic workup and results, Treatments administered, Consultant recommendations, Patient response to treatment, Procedures performed during stay]
Diagnostic Results:
Labs: [Abnormal and significant normal laboratory findings with dates]
Imaging: [All imaging studies with key findings and interpretations]
Discharge Medications:
[Complete list of medications to continue after discharge with dosages and frequencies]
Medications Discontinued:
[List of medications stopped at discharge with reasons if provided]
Discharge Disposition:
[Where patient is being discharged to]
Activity/Diet/Code Status:
[Document all specified]
Follow-up Plans:
[All scheduled follow-up appointments and instructions]
Allergies:
[Documented allergies or "No Known Allergies"]
Discharge Instructions:
[Any additional patient instructions or education provided]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided discharge summary into the exact structure below. 
Use ONLY information explicitly stated in the text. Do not infer, interpret, or
add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Copy exact medical terminology from source text
- Include all numerical values (labs, vitals, dosages)
- Preserve all dates and timelines
- Document both normal and abnormal findings
- Include all consultant recommendations verbatim
- List every medication with complete sig information
- Capture all follow-up instructions precisely

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)",
)

prompt = generate_prompt_for_discharge()

# Execution Placeholder for demonstration
llm_output = '```json\n{"extracted_clinical_data": "Patient Information: Jane Smith, MRN 654321, Discharge Date 2023-10-25...\\nAdmission Diagnoses: Pneumonia\\nDischarge Diagnoses: Resolved Pneumonia\\n..."}\n```'

# Cell 9: Clean LLM output and parse JSON (Discharge)
def process_llm_output_discharge_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted Discharge Summary Data ---")
        print(extracted_text)
        print("--------------------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_discharge_json(llm_output)
