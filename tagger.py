import pickle
from dataclasses import dataclass

import faiss
import numpy as np
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

@dataclass
class Tag:
    start: int
    end: int
    text: str
    cui: str
    name: str
    semtypes: list
    score: float
    matched: str   # the UMLS string that matched


class UMLSTagger:
    def __init__(self, index_path="umls.faiss", meta_path="umls_meta.pkl",
                 threshold=0.80, k=10, max_ngram=5, use_ner=True):
        self.index = faiss.read_index(index_path)
        self.index.hnsw.efSearch = 128
        with open(meta_path, "rb") as f:
            m = pickle.load(f)
        self.strings, self.cuis, self.pref, self.sty = m["strings"], m["cuis"], m["pref"], m["sty"]
        self.threshold, self.k, self.max_ngram = threshold, k, max_ngram

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        self.nlp = spacy.load("en_core_sci_sm")
        self.use_ner = use_ner

    @torch.no_grad()
    def _embed(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True,
                         max_length=32, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype("float32")

    def _candidates(self, doc):
        """Candidate mention spans as (char_start, char_end)."""
        spans = set()
        if self.use_ner:                       # scispaCy entities + noun chunks
            for e in doc.ents:
                spans.add((e.start_char, e.end_char))
            for nc in doc.noun_chunks:
                spans.add((nc.start_char, nc.end_char))
        # n-grams that start/end on content words, as a recall backstop
        toks = [t for t in doc if not t.is_space]
        for i, t in enumerate(toks):
            if t.is_stop or t.is_punct:
                continue
            for n in range(1, self.max_ngram + 1):
                if i + n > len(toks):
                    break
                last = toks[i + n - 1]
                if last.is_stop or last.is_punct:
                    continue
                spans.add((t.idx, last.idx + len(last)))
        return sorted(spans)

    def tag(self, text):
        doc = self.nlp(text)
        spans = self._candidates(doc)
        if not spans:
            return []
        mentions = [text[s:e] for s, e in spans]
        D, I = self.index.search(self._embed([m.lower() for m in mentions]), self.k)

        hits = []
        for (s, e), men, d_row, i_row in zip(spans, mentions, D, I):
            best = {}                          # best score per CUI for this span
            for score, idx in zip(d_row, i_row):
                if idx < 0 or score < self.threshold:
                    continue
                cui = self.cuis[idx]
                if cui not in best or score > best[cui][0]:
                    best[cui] = (float(score), self.strings[idx])
            if not best:
                continue
            cui, (score, matched) = max(best.items(), key=lambda kv: kv[1][0])
            hits.append(Tag(s, e, men, cui, self.pref.get(cui, matched),
                            [n for _, n in self.sty.get(cui, [])], score, matched))
        return self._resolve_overlaps(hits)

    @staticmethod
    def _resolve_overlaps(hits):
        """Greedy: prefer higher score, then longer span; drop overlapping spans."""
        hits = sorted(hits, key=lambda h: (-h.score, -(h.end - h.start)))
        chosen = []
        for h in hits:
            if all(h.end <= c.start or h.start >= c.end for c in chosen):
                chosen.append(h)
        return sorted(chosen, key=lambda h: h.start)


if __name__ == "__main__":
    tagger = UMLSTagger()
    note = ("Patient with type 2 diabetes mellitus and hypertension presented with "
            "chest pain. Started on metformin 500 mg; ECG showed no acute changes.")
    for t in tagger.tag(note):
        print(f"{t.text!r:35} {t.start}-{t.end} {t.cui}  {t.name}  [{', '.join(t.semtypes)}]  {t.score:.2f}")