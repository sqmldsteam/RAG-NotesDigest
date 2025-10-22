# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm
import os

# --- YOU MUST DEFINE full_rawdata HERE ---
full_rawdata = "Consultation for Ms. Doe on 2024-01-15. Reason: Altered mental status. Findings: Cardiac exam was WNL. Assessment: Likely metabolic encephalopathy. Recommendations: Check ammonia level and start lactulose."

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
    batch_chunks = chunks[i:i+batch_chunks]
    batch_emb = model.encode(batch_chunks)
    embeddings.append(batch_emb)

embeddings = np.vstack(embeddings)
print(f"Embedding dimension: {embeddings.shape[1]}")

# Cell 5: FAISS index and Execution
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))
print(f"FAISS index contains {index.ntotal} vectors")

# ----------------- USE CASE 3: CONSULT SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (Consultation Note)
def retrieve_rag_consult(query, top_k=4):
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for Consultation Note
def generate_prompt_for_consult():
    """Generates the specific prompt for extracting Consultation Note data."""
    retrieved_text = retrieve_rag_consult(
        "Extract specialist consultation findings, assessment, and recommendations."
    )
    
    required_structure = """
Consultation Information:
[Name, MRN, DOB, Age, Sex, Attending, Consulting Specialist, Consult Date, Reason for Consult]
History of Present Illness:
[Detailed narrative of current illness from consultant's perspective]
Past Medical History:
[Relevant chronic conditions for this consultation]
Past Surgical History:
[Relevant surgical history for this consultation]
Current Medications:
[Medications relevant to consultation]
Consultation Findings:
Physical Exam: [Specialist's focused examination findings]
Neurologic Exam: [If applicable - detailed neurologic assessment]
Other Systems: [Relevant system-based findings]
Diagnostic Review:
Labs: [Relevant laboratory findings reviewed]
Imaging: [Relevant imaging studies reviewed and interpreted]
Assessment:
[Consultant's clinical impressions and diagnoses]
Recommendations:
[Specific, actionable recommendations for primary team]
Plan:
[Detailed management plan from consultant]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided consultation note into the exact structure below. 
Use ONLY information explicitly stated in the text. Do not infer, interpret, or
add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Focus on consultant-specific findings and recommendations
- Copy exact medical terminology from source text
- Include all specialist exam findings verbatim
- Preserve all diagnostic interpretations
- Document all recommendations precisely
- Capture medication recommendations exactly
- Maintain consultant's clinical reasoning

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="https://api.groq.com/openai/v1",
)

prompt = generate_prompt_for_consult()

# Execution Placeholder for demonstration
llm_output = '```json\n{"extracted_clinical_data": "Consultation Information: Ms. Doe, Consult Date 2024-01-15, Reason for Consult Altered mental status.\\nConsultation Findings: Cardiac exam was WNL\\nRecommendations: Check ammonia level and start lactulose."}\n```'

# Cell 9: Clean LLM output and parse JSON (Consult)
def process_llm_output_consult_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted Consultation Note Data ---")
        print(extracted_text)
        print("--------------------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_consult_json(llm_output)
