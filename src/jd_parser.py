"""
jd_parser.py
============
Encodes the Senior AI Engineer JD into structured rules.

This file does NOT read job_description.docx at runtime. We read the JD once
as humans, and hand-encode what it means into this module. This is a
deliberate design choice: the JD is a single fixed document for this
challenge, not something we need to re-parse generically every run. Hand
-encoding also makes the logic auditable and defensible in interview --
every disqualifier here maps to a specific sentence in the JD.

If this were a production system ranking against MANY different JDs, you'd
want an LLM or NLP pipeline to extract these fields automatically. We are
explicit that we did NOT do that here, and explain why in the README.
"""

from dataclasses import dataclass, field


# Consulting-only firms the JD explicitly flags as a soft disqualifier
# (unless candidate also has prior product-company experience).
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini",
}

# Keywords signalling production embeddings / retrieval experience.
RETRIEVAL_KEYWORDS = {
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embedding", "embeddings", "vector search",
    "semantic search", "dense retrieval", "retrieval",
}

VECTOR_DB_KEYWORDS = {
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector db", "hybrid search",
}

EVAL_FRAMEWORK_KEYWORDS = {
    "ndcg", "mrr", "map", "a/b test", "ab test", "offline-to-online",
    "offline to online", "evaluation framework", "precision@k", "recall@k",
}

LANGCHAIN_WRAPPER_KEYWORDS = {
    "langchain", "llamaindex", "openai api", "gpt-4", "gpt4", "chatgpt api",
}

NLP_IR_KEYWORDS = {
    "nlp", "natural language processing", "information retrieval", "ir",
    "search", "ranking", "recommendation", "recommender", "text",
    "language model", "llm",
}

CV_SPEECH_ROBOTICS_KEYWORDS = {
    "computer vision", "cv pipeline", "image classification", "object detection",
    "speech recognition", "asr", "robotics", "slam", "autonomous",
}

TIER1_CITIES_PREFERRED = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram"}

# What the JD literally asks for, used for the soft/semantic side later.
JD_SEMANTIC_TEXT = """
Senior AI Engineer, founding team, Redrob AI. Own the intelligence layer:
ranking, retrieval, and matching systems for candidate-JD search.
Production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users,
including embedding drift, index refresh, retrieval-quality regression.
Production experience with vector databases or hybrid search infrastructure
(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python and code quality. Hands-on evaluation framework design for
ranking systems: NDCG, MRR, MAP, offline-to-online correlation, A/B testing.
Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank
(XGBoost, neural), HR-tech/recruiting/marketplace background, distributed
systems, large-scale inference, open-source contributions.
Ideal candidate has shipped an end-to-end ranking, search, or recommendation
system to real users at meaningful scale, has opinions on hybrid vs dense
retrieval and offline vs online evaluation, located in or willing to
relocate to Noida or Pune.
"""


@dataclass
class JDRequirements:
    min_years: float = 5.0
    max_years: float = 9.0  # soft ceiling, not a hard cutoff -- JD says band is flexible
    preferred_cities: set = field(default_factory=lambda: TIER1_CITIES_PREFERRED)
    semantic_text: str = JD_SEMANTIC_TEXT.strip()


def get_jd_requirements() -> JDRequirements:
    return JDRequirements()


def text_blob_for_candidate(career_history_text: str, keyword_set: set) -> bool:
    """Utility: does any keyword in keyword_set appear in the given text?"""
    text = career_history_text.lower()
    return any(kw in text for kw in keyword_set)
