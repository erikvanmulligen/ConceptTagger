import pickle
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

META = Path("/Users/mulligen/data/umls/2026AA/META")
MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# Restrict vocabularies to keep the index manageable (adjust to taste)
SABS = {"SNOMEDCT_US", "MSH", "RXNORM", "ICD10CM", "NCI", "OMIM"}
# Optional: keep only some semantic types (TUIs), e.g. {"T047","T184","T121"}; None = keep all
KEEP_TUIS = {
    "T047",  # Disease or Syndrome
    "T184",  # Sign or Symptom
    "T121",  # Pharmacologic Substance
    "T167",  # Substance
    "T190",  # Anatomical Abnormality
    "T017",  # Anatomical Structure
    "T087",  # Amino Acid Sequence
    "T116",  # Amino Acid, Peptide, or Protein
    "T029",  # Body Location or Region
    "T033",  # Finding
}

# --- Parse MRSTY: CUI -> semantic types
sty = defaultdict(set)
with open(META / "MRSTY.RRF", encoding="utf-8") as f:
    for line in f:
        p = line.split("|")
        sty[p[0]].add((p[1], p[3]))  # (TUI, type name)

# --- Parse MRCONSO: English, non-suppressed strings
pairs = set()          # (lowercased string, CUI)
pref = {}              # CUI -> preferred name
with open(META / "MRCONSO.RRF", encoding="utf-8") as f:
    for line in f:
        p = line.split("|")
        cui, lat, ts, stt, ispref, sab, text, suppress = p[0], p[1], p[2], p[4], p[6], p[11], p[14], p[16]
        if lat != "ENG" or suppress != "N" or sab not in SABS:
            continue
        if KEEP_TUIS and not ({t for t, _ in sty[cui]} & KEEP_TUIS):
            continue
        if len(text) < 3 or len(text) > 100:
            continue
        pairs.add((text.lower(), cui))
        if ts == "P" and stt == "PF" and ispref == "Y":
            pref[cui] = text

strings, cuis = zip(*sorted(pairs))
strings, cuis = list(strings), list(cuis)
print(f"{len(strings):,} strings, {len(set(cuis)):,} concepts")

# --- Embed
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()
if device == "cuda" or device == "mps":
    model.half()

@torch.no_grad()
def embed(texts, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        h = model(**b).last_hidden_state[:, 0]          # [CLS] as in SapBERT
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype("float32")


# --- Index (HNSW, inner product == cosine on normalized vectors)
index = faiss.IndexHNSWFlat(768, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 100
CHUNK = 100_000
for i in range(0, len(strings), CHUNK):
    index.add(embed(strings[i:i + CHUNK]))
    print(f"indexed {min(i + CHUNK, len(strings)):,}")

faiss.write_index(index, "umls.faiss")
with open("umls_meta.pkl", "wb") as f:
    pickle.dump({"strings": strings, "cuis": cuis, "pref": pref,
                 "sty": {c: sorted(sty[c]) for c in set(cuis)}}, f)