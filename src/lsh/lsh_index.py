"""
Locality-Sensitive Hashing (LSH) Index implementation for fast sub-linear textual similarity search.
Uses datasketch MinHash with optional exact Jaccard reranking and global candidate backfilling.
"""

import logging
from typing import Iterable, List, Tuple, Dict, Optional, Union
from datasketch import MinHash, MinHashLSH

logging.basicConfig(level=logging.INFO)


class LSHIndex:
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.5,
        tokenizer=None,
    ):
        self.num_perm = int(num_perm)
        self.threshold = float(threshold)
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # In-memory caches
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_token_sets: Dict[str, frozenset] = {}
        self.doc_minhash: Dict[str, MinHash] = {}

        # Tokenizer
        self.tokenizer = tokenizer if tokenizer else self.default_tokenizer

    @staticmethod
    def default_tokenizer(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return [t for t in text.lower().split() if t]

    def _to_tokens(self, text_or_tokens) -> List[str]:
        if isinstance(text_or_tokens, str):
            return self.tokenizer(text_or_tokens)
        try:
            return [str(t) for t in text_or_tokens]
        except Exception:
            return []

    def _minhash_from_tokens(self, tokens: Iterable[str]) -> Optional[MinHash]:
        tokens = list(tokens)
        if not tokens:
            return None
        m = MinHash(num_perm=self.num_perm)
        for t in tokens:
            m.update(t.encode("utf8"))
        return m

    def add_document(self, doc_id: str, text_or_tokens) -> None:
        if doc_id in self.doc_tokens:
            logging.info(f"Duplicate doc_id '{doc_id}' – skipping.")
            return

        tokens = self._to_tokens(text_or_tokens)
        if not tokens:
            logging.warning(f"Document '{doc_id}' has no tokens – skipping.")
            return

        m = self._minhash_from_tokens(tokens)
        if m is None:
            logging.warning(f"Document '{doc_id}' produced empty MinHash – skipping.")
            return

        self.lsh.insert(doc_id, m)
        self.doc_tokens[doc_id] = tokens
        self.doc_token_sets[doc_id] = frozenset(tokens)
        self.doc_minhash[doc_id] = m

    def add_documents_batch(self, docs: Iterable[Tuple[str, Union[Iterable[str], str]]]) -> None:
        for doc_id, text_or_tokens in docs:
            self.add_document(doc_id, text_or_tokens)

    def add_document_minhash(self, doc_id: str, m: MinHash, tokens: Optional[Iterable[str]] = None) -> None:
        if doc_id in self.doc_tokens or doc_id in self.doc_minhash:
            logging.info(f"Duplicate doc_id '{doc_id}' – skipping.")
            return
        if not isinstance(m, MinHash):
            raise TypeError("m must be a datasketch.MinHash instance")

        self.lsh.insert(doc_id, m)
        if tokens is not None:
            tok_list = self._to_tokens(tokens)
            self.doc_tokens[doc_id] = tok_list
            self.doc_token_sets[doc_id] = frozenset(tok_list)
        self.doc_minhash[doc_id] = m

    def remove_document(self, doc_id: str) -> None:
        self.doc_tokens.pop(doc_id, None)
        self.doc_token_sets.pop(doc_id, None)
        self.doc_minhash.pop(doc_id, None)

    def size(self) -> int:
        return len(self.doc_minhash)

    def _minhash_query_candidates(self, q_m: MinHash) -> List[str]:
        try:
            return list(self.lsh.query(q_m))
        except Exception as e:
            logging.warning(f"LSH query failed: {e}")
            return []

    def _global_candidates_by_minhash(self, q_m: MinHash, budget: Optional[int] = None):
        pairs = [(doc_id, float(q_m.jaccard(m))) for doc_id, m in self.doc_minhash.items()]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs if budget is None else pairs[:budget]

    def _exact_rerank(
        self,
        q_tokens: Iterable[str],
        candidates_scored: List[Tuple[str, float]],
        exact_pool: int
    ) -> List[Tuple[str, float]]:
        q_set = frozenset(q_tokens)
        pool = candidates_scored[:min(exact_pool, len(candidates_scored))]
        reranked: List[Tuple[str, float]] = []

        for doc_id, approx_sim in pool:
            dset = self.doc_token_sets.get(doc_id)
            if dset is None:
                reranked.append((doc_id, approx_sim))
                continue
            inter = len(q_set & dset)
            union = len(q_set) + len(dset) - inter
            sim_exact = 0.0 if union == 0 else inter / union
            reranked.append((doc_id, float(sim_exact)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def query_similar(
        self,
        query_text: str,
        top_n: int = 3,
        rerank_exact: bool = False,
        exact_pool: int = 20,
        min_candidates: int = 0,
        backfill: bool = True,
        backfill_budget: Optional[int] = None,
        return_pairs: bool = True,
    ):
        tokens = self._to_tokens(query_text)
        if not tokens:
            logging.warning("Query has no tokens.")
            return []

        q_m = self._minhash_from_tokens(tokens)
        if q_m is None:
            logging.warning("Query MinHash is empty.")
            return []

        cand_ids = self._minhash_query_candidates(q_m)
        approx_scored = [(doc_id, float(q_m.jaccard(self.doc_minhash[doc_id])))
                         for doc_id in cand_ids]
        approx_scored.sort(key=lambda x: x[1], reverse=True)

        if backfill:
            need = max(0, min_candidates - len(approx_scored))
            if need > 0:
                budget = backfill_budget if backfill_budget is not None else max(need * 3, top_n * 20)
                global_scored = self._global_candidates_by_minhash(q_m, budget=budget)
                seen = {doc_id for doc_id, _ in approx_scored}
                for doc_id, score in global_scored:
                    if doc_id not in seen:
                        approx_scored.append((doc_id, score))
                        seen.add(doc_id)
                    if len(approx_scored) >= min_candidates:
                        break
                approx_scored.sort(key=lambda x: x[1], reverse=True)

        if rerank_exact and approx_scored:
            reranked = self._exact_rerank(tokens, approx_scored, exact_pool=exact_pool)
            out = reranked[:top_n]
        else:
            out = approx_scored[:top_n]

        return out if return_pairs else [doc_id for doc_id, _ in out]
