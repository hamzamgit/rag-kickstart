"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

For every document, this module builds a tree from the existing chunk leaves:

    level 3  [root summary]
    level 2  [summary] [summary]
    level 1  [summary] [summary] [summary]
    level 0  [chunk] [chunk] [chunk] [chunk]

The tree preserves parent/child links in PostgreSQL. Retrieval does *not* walk
one branch of the tree: it searches every level as one flat pgvector pool, so
the top-k results may contain a specific chunk, a cluster summary, or a root.
"""

import argparse
from collections import defaultdict

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.mixture import GaussianMixture

from app.llm import build_llm
from app.indexing.raptor.raptor_config import RaptorSettings, load_raptor_settings
from app.indexing.raptor.raptor_retriever import build_raptor_retriever


SUMMARY_TEMPLATE = """Write one concise factual summary of these related nodes
from the document "{title}" (topic: {topic}).

Nodes to summarize:
---
{node_contents}
---

The summary must preserve important technical details and only use information
from the supplied nodes. Write plain prose; do not use a heading or bullets.
"""

ANSWER_TEMPLATE = """Answer the user's question using only the RAPTOR nodes below.

Question:
{question}

Retrieved RAPTOR nodes:
---
{context}
---

Use the detailed chunks for precise claims and the summaries for document-level
context. If the nodes do not support an answer, say so clearly. Write 3 to 5
sentences without bullets, tables, or code blocks.
"""

summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

# 16,000 characters are roughly 4,000 input tokens for English prose. Together
# with the 600-token completion limit below, this stays safely below Groq's
# 8,000-token-per-request limit.
MAX_SUMMARY_INPUT_CHARACTERS = 16_000
MAX_ANSWER_CONTEXT_CHARACTERS = 18_000


def format_docs(docs) -> str:
    """Label node type and level so the answer model can use each appropriately."""
    sections = []
    remaining_characters = MAX_ANSWER_CONTEXT_CHARACTERS

    for doc in docs:
        heading = f"[{doc.metadata['node_type']} node, level {doc.metadata['level']}]\n"
        available_content_characters = remaining_characters - len(heading)

        if available_content_characters <= 0:
            break

        section = heading + doc.page_content[:available_content_characters]
        sections.append(section)
        remaining_characters -= len(section)

    return "\n\n".join(sections)


def group_nodes_with_gmm(nodes: list[dict], settings: RaptorSettings) -> list[list[dict]]:
    """Cluster a tree level with GMM, selecting components by the lowest BIC.

    The supplied minimum and maximum cluster settings define the candidate range.
    BIC selects the best candidate for the embeddings rather than hard-coding a
    cluster count. Two or fewer nodes are summarized directly into one parent.
    """
    if len(nodes) <= settings.min_clusters:
        return [nodes]

    vectors = np.asarray([np.asarray(node["embedding"], dtype=float) for node in nodes])
    maximum_components = min(settings.max_clusters, len(nodes) - 1)
    candidate_counts = range(settings.min_clusters, maximum_components + 1)

    models = []
    for component_count in candidate_counts:
        model = GaussianMixture(
            n_components=component_count,
            covariance_type="diag",
            random_state=42,
            n_init=3,
        )
        model.fit(vectors)
        models.append(model)

    best_model = min(models, key=lambda model: model.bic(vectors))
    labels = best_model.predict(vectors)
    groups_by_label = defaultdict(list)

    for node, label in zip(nodes, labels):
        groups_by_label[int(label)].append(node)

    return [groups_by_label[label] for label in sorted(groups_by_label)]


def summarize_nodes(llm, document: dict, nodes: list[dict]) -> str:
    """Create the text stored in one RAPTOR summary or root node."""
    # Give each member of a cluster an equal share of the prompt budget. This
    # keeps a large final/root cluster from exceeding the provider token limit.
    characters_per_node = max(1, MAX_SUMMARY_INPUT_CHARACTERS // len(nodes))
    node_contents = "\n\n---\n\n".join(
        node["content"][:characters_per_node] for node in nodes
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    return summary_chain.invoke(
        {
            "title": document["title"],
            "topic": document["topic"],
            "node_contents": node_contents,
        }
    )


def build_tree_for_document(
    llm,
    raptor_retriever,
    document: dict,
    settings: RaptorSettings,
) -> int:
    """Copy chunk leaves, recursively cluster/summarize, and end at one root."""
    source_chunks = raptor_retriever.get_document_chunks(document["document_id"])
    if not source_chunks:
        print(f"Skipping {document['source_filename']}: no source chunks found.")
        return 0

    # A document with a failed earlier attempt has no root. Remove any partial
    # nodes so the reconstructed tree is complete and internally consistent.
    raptor_retriever.clear_document_tree(document["document_id"])
    current_nodes = raptor_retriever.insert_leaf_nodes(
        document["document_id"], source_chunks
    )
    node_count = len(current_nodes)

    for level in range(1, settings.max_levels + 1):
        # The final allowed level always combines the remaining nodes into one
        # root, which honors RAPTOR_STOP_CONDITION=single_root.
        groups = (
            [current_nodes]
            if level == settings.max_levels
            else group_nodes_with_gmm(current_nodes, settings)
        )

        next_level_nodes = []
        for cluster_index, group in enumerate(groups):
            summary = summarize_nodes(llm, document, group)
            summary_vector = raptor_retriever.embeddings.embed_query(summary)
            node_type = "root" if len(groups) == 1 else "summary"
            summary_node = raptor_retriever.insert_summary_node(
                document_id=document["document_id"],
                parent_node_id=None,
                level=level,
                node_type=node_type,
                cluster_index=cluster_index,
                content=summary,
                embedding=summary_vector,
                child_node_ids=[node["node_id"] for node in group],
            )
            next_level_nodes.append(summary_node)
            node_count += 1

        if len(next_level_nodes) == 1:
            break

        current_nodes = next_level_nodes

    return node_count


def build_missing_raptor_trees() -> int:
    """Build RAPTOR trees only for documents that do not already have a root."""
    settings = load_raptor_settings()
    # RAPTOR summaries are intentionally concise; avoid the default 8,192-token
    # completion reservation that caused Groq 413 request-size failures.
    llm = build_llm(max_tokens=600)
    raptor_retriever = build_raptor_retriever()
    raptor_retriever.ensure_schema()
    documents = raptor_retriever.documents_without_root()

    if not documents:
        print("Every stored document already has a completed RAPTOR tree.")
        return 0

    for number, document in enumerate(documents, start=1):
        print(
            f"Building RAPTOR tree {number}/{len(documents)}: "
            f"{document['source_filename']}"
        )
        node_count = build_tree_for_document(
            llm=llm,
            raptor_retriever=raptor_retriever,
            document=document,
            settings=settings,
        )
        print(f"Stored {node_count} RAPTOR nodes for {document['title']}.")

    return len(documents)


def answer_with_raptor(question: str, retriever) -> str:
    """Search the complete RAPTOR node pool, then answer from the top-k nodes."""
    llm = build_llm(max_tokens=600)

    # One normal pgvector search over leaves, cluster summaries, and roots.
    docs = retriever.invoke(question)
    context = format_docs(docs)

    print("\nRetrieved RAPTOR nodes:")
    for document in docs:
        print(
            f"- {document.metadata['node_type']} "
            f"(level {document.metadata['level']}): "
            f"{document.page_content[:180].replace(chr(10), ' ')}"
        )

    if not context:
        return "No RAPTOR nodes are indexed yet. Build the missing trees first."

    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain.invoke({"question": question, "context": context})


def main():
    parser = argparse.ArgumentParser(description="Build and query local RAPTOR trees.")
    parser.add_argument(
        "--build-missing",
        action="store_true",
        help="Build trees only for documents that do not already have a root.",
    )
    args = parser.parse_args()

    if args.build_missing:
        build_missing_raptor_trees()
        return

    raptor_retriever = build_raptor_retriever(k=8)
    raptor_retriever.ensure_schema()
    question = "How does Self-RAG decide whether retrieval is useful?"
    final_answer = answer_with_raptor(question, raptor_retriever)

    print("\nFinal answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
