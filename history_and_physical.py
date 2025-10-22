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

# ----------------- USE CASE 1: H&P SPECIFIC CODE -----------------

# Cell 6: RAG retrieval function (H&P)
def retrieve_rag_hp(query, top_k=5): # Increased k for comprehensive note
    """Retrieves relevant chunks based on a query using the FAISS index."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator for H&P
def generate_prompt_for_hp():
    """Generates the specific prompt for extracting History & Physical data."""
    retrieved_text = retrieve_rag_hp(
        "Extract all components of a History and Physical examination note including subjective, objective, assessment, and plan."
    )
    
    required_structure = """
Patient Information:
[Name, MRN, DOB, Age, Sex, PCP, Attending, Admit Date, Location, Information Source]
Chief Complaint & History of Present Illness:
[Primary reason for admission and detailed narrative of current illness exactly as documented]
Past Medical History:
[Comprehensive list of ALL chronic conditions with relevant dates if provided]
Past Surgical History:
[Detailed surgical procedures with dates, laterality, and surgeons if documented]
Current Medications:
[Complete list of pre-admission medications with exact dosages, frequencies, and instructions]
Allergies:
[Documented allergies or "No Known Allergies"]
Social History:
[Living situation, tobacco/alcohol/substance use, social support exactly as documented]
Family History:
[Relevant family medical history or "Unremarkable"]
Review of Systems:
[System-by-system documentation of positive and negative findings verbatim]
Physical Examination:
Vital Signs: [Current and relevant vitals with values]
General: [Appearance, behavior, mental status]
HEENT: [Head, eyes, ears, nose, throat findings]
Neck: [Range of motion, lymph nodes, JVD]
Respiratory: [Breath sounds, respiratory effort]
Cardiovascular: [Heart sounds, rhythm, murmurs]
Abdomen: [Bowel sounds, tenderness, masses]
Musculoskeletal: [Strength, range of motion, pulses]
Skin: [Rashes, lesions, wounds]
Extremities: [Edema, pulses]
Neurologic: [Mental status, cranial nerves, motor/sensory function]
Diagnostic Results:
Labs: [Relevant abnormal laboratory findings with values and dates]
Imaging: [Key imaging results and interpretations with dates]
Assessment & Plan:
[Problem-based list with specific diagnostic and therapeutic plans for each active issue exactly as documented]
    """
    
    prompt = f"""
You are a medical documentation specialist. Extract and organize ALL clinical
information from the provided History & Physical note into the exact
structure below. Use ONLY information explicitly stated in the text. Do not
infer, interpret, or add any information not present in the source document.

Return the result as a STRICT JSON object with a single key 'extracted_clinical_data' 
whose value is the formatted output matching the structure below.

REQUIRED OUTPUT STRUCTURE (Fill in the data next to the colon and brackets):
{required_structure}

KEY EXTRACTION RULES:
- Copy exact medical terminology from source text
- Include all numerical values (labs, vitals, dosages)
- Preserve all dates and timelines
- Document both normal and abnormal findings
- Include complete medication sig information
- Capture all physical exam findings verbatim
- Maintain problem list exactly as documented

Source Text:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM and Execution
client = OpenAI(
    api_key='',
    base_url="https://api.groq.com/openai/v1",
)

prompt = generate_prompt_for_hp()

# For a full run, uncomment the next two lines:
# response = client.responses.create(
#     model="openai/gpt-oss-20b",
#     input=prompt
# )
# llm_output = response.output_text.strip()
llm_output = '```json\n{"extracted_clinical_data": "Patient Information: John Doe, MRN 123456, DOB N/A, Age N/A, Sex N/A, PCP N/A, Attending N/A, Admit Date 2023-10-20, Location N/A, Information Source N/A\\nChief Complaint & History of Present Illness: Severe headache for N/A. Narrative: N/A\\n..."}\n```'


# Cell 9: Clean LLM output and parse JSON (H&P)
def process_llm_output_hp_json(llm_output):
    """Cleans and parses the LLM JSON output and prints the extracted text."""
    # Remove Markdown-style backticks if present
    if llm_output.startswith("```") and llm_output.endswith("```"):
        llm_output = llm_output.strip("```").strip()
        # Handle common case of ```json ... ```
        if llm_output.startswith("json"):
            llm_output = llm_output[4:].strip()

    try:
        data_json = json.loads(llm_output)
        extracted_text = data_json.get("extracted_clinical_data", "Error: Key not found.")
        print("--- Extracted H&P Clinical Data ---")
        print(extracted_text)
        print("-----------------------------------")
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM output:")
        print(str(e))
        print("Raw LLM output:")
        print(llm_output)

process_llm_output_hp_json(llm_output)
