# Cell 1: Imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from tqdm.auto import tqdm


# Cell 3: Chunking function
def chunk_text(text, chunk_size=800):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(full_rawdata)
print(f"Total chunks created: {len(chunks)}")

# Cell 4: Embeddings using Hugging Face (fast)
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

# Cell 5: FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))
print(f"FAISS index contains {index.ntotal} vectors")

# Cell 6: RAG retrieval function
def retrieve_rag(query, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

# Cell 7: Prompt generator
def generate_prompt_for_demographics():
    retrieved_text = retrieve_rag(
        "Extract patient demographics including name, MRN, admission date, location, and date of service"
    )
    
    prompt = f"""
You are a medical documentation assistant. Extract ONLY the patient demographics from the following note and return strictly as JSON with the following keys:
- patient_first_name
- patient_last_name
- patient_name
- patient_mrn
- location
- admission_date
- date_of_service

Use only information present in the note. If a field is missing, use an empty string.

Medical Note:
{retrieved_text}
"""
    return prompt

# Cell 8: Call LLM # Other models can be used
client = OpenAI(
    api_key='',
    base_url="https://api.groq.com/openai/v1",
)

prompt = generate_prompt_for_demographics()

response = client.responses.create(
    model="openai/gpt-oss-20b",
    input=prompt
)

# Cell 9: Clean LLM output and parse JSON
llm_output = response.output_text.strip()

# Remove Markdown-style backticks if present
if llm_output.startswith("```") and llm_output.endswith("```"):
    llm_output = llm_output.strip("```").strip()

try:
    demographics_json = json.loads(llm_output)
    print("Extracted Demographics JSON:")
    print(json.dumps(demographics_json, indent=4))
except json.JSONDecodeError as e:
    print("Failed to parse JSON from LLM output:")
    print(str(e))
    print("Raw LLM output:")
    print(llm_output)
