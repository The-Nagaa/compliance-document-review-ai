"""
Vector Store Repository Abstraction.

Supports in-memory & persistent local vector operations with exact cosine indexing.
Easily exportable to PostgreSQL + pgvector.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ai.app.core.config import settings
from ai.app.core.logging import logger
from ai.app.models.entities import Disclosure, Precedent, Rule
from ai.app.services.embedding_service import embedding_service


class VectorStore:
    """
    In-memory vector store with JSON persistence and pgvector schema compatibility.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.VECTOR_STORE_PATH
        self.rules: Dict[str, Rule] = {}
        self.disclosures: Dict[str, Disclosure] = {}
        self.precedents: Dict[str, Precedent] = {}

    def clear(self) -> None:
        """Clear all collections."""
        self.rules.clear()
        self.disclosures.clear()
        self.precedents.clear()

    # --- Rules Collection ---
    def add_rule(self, rule: Rule) -> None:
        if rule.embedding is None:
            rule.embedding = embedding_service.get_embedding(rule.text)
        self.rules[rule.id] = rule

    def add_rules_batch(self, rules: List[Rule]) -> None:
        for r in rules:
            self.add_rule(r)

    def search_rules(
        self, query_vector: List[float], top_k: int = 5, category: Optional[str] = None
    ) -> List[Tuple[Rule, float]]:
        results: List[Tuple[Rule, float]] = []
        for rule in self.rules.values():
            if category and rule.category != category:
                continue
            if rule.embedding:
                sim = embedding_service.cosine_similarity(query_vector, rule.embedding)
                results.append((rule, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_rules(self) -> List[Rule]:
        return list(self.rules.values())

    # --- Disclosures Collection ---
    def add_disclosure(self, disclosure: Disclosure) -> None:
        if disclosure.embedding is None:
            disclosure.embedding = embedding_service.get_embedding(disclosure.text)
        self.disclosures[disclosure.id] = disclosure

    def add_disclosures_batch(self, disclosures: List[Disclosure]) -> None:
        for d in disclosures:
            self.add_disclosure(d)

    def search_disclosures(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[Disclosure, float]]:
        results: List[Tuple[Disclosure, float]] = []
        for d in self.disclosures.values():
            if d.embedding:
                sim = embedding_service.cosine_similarity(query_vector, d.embedding)
                results.append((d, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_disclosures(self) -> List[Disclosure]:
        return list(self.disclosures.values())

    # --- Precedents Collection ---
    def add_precedent(self, precedent: Precedent) -> None:
        if precedent.embedding is None:
            precedent.embedding = embedding_service.get_embedding(precedent.masked_text)
        self.precedents[precedent.document_id] = precedent

    def add_precedents_batch(self, precedents: List[Precedent]) -> None:
        for p in precedents:
            self.add_precedent(p)

    def search_precedents(
        self, query_vector: List[float], top_k: int = 3
    ) -> List[Tuple[Precedent, float]]:
        results: List[Tuple[Precedent, float]] = []
        for p in self.precedents.values():
            if p.embedding:
                sim = embedding_service.cosine_similarity(query_vector, p.embedding)
                results.append((p, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_precedents(self) -> List[Precedent]:
        return list(self.precedents.values())

    # --- Persistence ---
    def save_to_disk(self, filepath: Optional[Path] = None) -> None:
        target_file = filepath or self.storage_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "rules": [
                {
                    "id": r.id,
                    "text": r.text,
                    "category": r.category,
                    "embedding": r.embedding,
                    "metadata": r.metadata,
                }
                for r in self.rules.values()
            ],
            "disclosures": [
                {
                    "id": d.id,
                    "text": d.text,
                    "type": d.type,
                    "embedding": d.embedding,
                    "metadata": d.metadata,
                }
                for d in self.disclosures.values()
            ],
            "precedents": [
                {
                    "document_id": p.document_id,
                    "masked_text": p.masked_text,
                    "decision": p.decision,
                    "comment": p.comment,
                    "embedding": p.embedding,
                    "metadata": p.metadata,
                }
                for p in self.precedents.values()
            ],
        }

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved vector store to {target_file} (rules={len(self.rules)}, disclosures={len(self.disclosures)}, precedents={len(self.precedents)})")

    def load_from_disk(self, filepath: Optional[Path] = None) -> bool:
        target_file = filepath or self.storage_path
        if not target_file.exists():
            return False

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.clear()

            for r_data in payload.get("rules", []):
                self.rules[r_data["id"]] = Rule(
                    id=r_data["id"],
                    text=r_data["text"],
                    category=r_data["category"],
                    embedding=r_data.get("embedding"),
                    metadata=r_data.get("metadata", {}),
                )

            for d_data in payload.get("disclosures", []):
                self.disclosures[d_data["id"]] = Disclosure(
                    id=d_data["id"],
                    text=d_data["text"],
                    type=d_data["type"],
                    embedding=d_data.get("embedding"),
                    metadata=d_data.get("metadata", {}),
                )

            for p_data in payload.get("precedents", []):
                self.precedents[p_data["document_id"]] = Precedent(
                    document_id=p_data["document_id"],
                    masked_text=p_data["masked_text"],
                    decision=p_data["decision"],
                    comment=p_data["comment"],
                    embedding=p_data.get("embedding"),
                    metadata=p_data.get("metadata", {}),
                )

            logger.info(f"Loaded vector store from {target_file} (rules={len(self.rules)}, disclosures={len(self.disclosures)}, precedents={len(self.precedents)})")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {target_file}: {e}")
            return False

    def export_pgvector_sql(self) -> str:
        """
        Generate SQL migration/seed script for PostgreSQL with pgvector extension.
        """
        sql_lines = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            "",
            "CREATE TABLE IF NOT EXISTS compliance_rules (",
            "    id VARCHAR(50) PRIMARY KEY,",
            "    text TEXT NOT NULL,",
            "    category VARCHAR(100),",
            f"    embedding vector({embedding_service.dimension})",
            ");",
            "",
            "CREATE TABLE IF NOT EXISTS compliance_disclosures (",
            "    id VARCHAR(50) PRIMARY KEY,",
            "    text TEXT NOT NULL,",
            "    type VARCHAR(100),",
            f"    embedding vector({embedding_service.dimension})",
            ");",
            "",
            "CREATE TABLE IF NOT EXISTS compliance_precedents (",
            "    document_id VARCHAR(100) PRIMARY KEY,",
            "    masked_text TEXT NOT NULL,",
            "    decision VARCHAR(50) NOT NULL,",
            "    officer_comment TEXT NOT NULL,",
            f"    embedding vector({embedding_service.dimension})",
            ");",
        ]
        return "\n".join(sql_lines)


vector_store = VectorStore()
