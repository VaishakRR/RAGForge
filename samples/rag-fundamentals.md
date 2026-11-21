# RAG Fundamentals

Retrieval-Augmented Generation (RAG) is a pattern that combines an information
retrieval system with a language model. Instead of relying only on parameters
memorized during training, the model receives relevant passages from a document
collection at answer time. This grounds the answer in sources that can be
inspected and cited.

## The RAG pipeline

A typical local RAG pipeline has six stages:

1. **Ingestion** — read documents and extract plain text. PDFs, Markdown, HTML,
   DOCX, and source code each need a different parser.
2. **Chunking** — split long documents into passages small enough to fit in the
   model's context window.
3. **Embedding** — convert each chunk into a dense vector using an embedding
   model such as `nomic-embed-text`.
4. **Indexing** — store vectors and metadata so they can be searched quickly.
5. **Retrieval** — embed the user's question and find the most relevant chunks.
6. **Generation** — place the retrieved chunks into a prompt and ask the model
   to answer using only those sources.

## Chunking with overlap

Chunks are usually 300 to 1200 characters or 100 to 400 tokens. A sliding
window with overlap is common: the next chunk starts before the previous one
ends, so a sentence that straddles a boundary still appears intact in at least
one chunk.

Overlap costs storage and embedding time, so it is a trade-off. A good default
is roughly 10 to 20 percent of the chunk size. Boundary-aware chunking tries to
cut at paragraph or sentence endings instead of mid-word, which improves both
embedding quality and readability of citations.

Very small chunks lose context; very large chunks dilute the embedding and waste
context window. The right size depends on the documents and the questions.

## Embeddings and cosine similarity

An embedding model maps text to a fixed-length vector, for example 768
dimensions for `nomic-embed-text`. Semantically similar passages end up close in
vector space. If vectors are L2-normalized, cosine similarity is simply their dot
product.

Dense retrieval is strong at matching meaning and paraphrases. It can miss exact
identifiers, error codes, and rare keywords, which is why hybrid retrieval is
often better.

## BM25 keyword retrieval

BM25 is a classic sparse ranking function. It scores a document for a query by
combining term frequency, inverse document frequency, and document length
normalization. Rare terms receive more weight, and very long documents are
penalized so they do not dominate simply by containing more words.

BM25 excels at exact matches: names, error messages, API symbols, and product
codes. It is cheap to compute and needs no training.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines several ranked lists without needing their
scores to be on the same scale. For each document, RRF sums a contribution from
every list it appears in:

    score(d) = sum over lists L of 1 / (k + rank_L(d))

The constant `k` is usually 60. A document ranked highly by both dense and BM25
retrieval rises to the top, while a document found by only one method can still
contribute. RRF is simple, robust, and a strong default for hybrid search.

## Maximal Marginal Relevance

MMR balances relevance and diversity when selecting the final chunks. It picks
the candidate that maximizes:

    lambda * similarity(query, candidate)
    - (1 - lambda) * max similarity(candidate, already selected)

With `lambda` near 1 the selection is relevance-focused; near 0 it is
diversity-focused. A common value is 0.5 to 0.7. MMR prevents several nearly
identical chunks from filling the context window.

## Grounding and citations

The generation prompt should instruct the model to answer only from the supplied
sources, to cite them inline as `[1]`, `[2]`, and so on, and to admit when the
sources are insufficient. Numbered sources let the interface show exactly which
passage supported each claim. Citations make hallucinations easier to detect.

## Evaluation

Useful retrieval metrics include recall@k, mean reciprocal rank (MRR), and
normalized discounted cumulative gain (nDCG). End-to-end evaluation should also
check answer faithfulness, citation correctness, and whether the system refuses
to answer when the documents do not contain the information.

## Common failure modes

- Chunks split in the middle of a key sentence.
- An embedding model that is weak for the document language or domain.
- Retrieving many near-duplicate chunks instead of diverse evidence.
- A prompt that lets the model answer from prior knowledge.
- Changing the embedding model without rebuilding the index, which makes query
  and document vectors incompatible.
