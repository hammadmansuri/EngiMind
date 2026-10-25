# Production RAG checklist

This is a practical production-readiness checklist for a retrieval-augmented generation system like this project. It is intentionally focused on the RAG layer, not just the model or UI.

## 1. Retrieval quality

- [ ] Chunking strategy is tuned for document structure, not just arbitrary size
- [ ] Overlap is controlled and justified
- [ ] Small but coherent chunks are preferred over fragmented text
- [ ] Chunks preserve section headers, page context, and source metadata
- [ ] Retrieval is tested on realistic questions, not only happy-path prompts
- [ ] Top-k retrieval is tuned for precision and recall trade-offs
- [ ] Duplicate and near-duplicate chunks are removed before ranking
- [ ] Ranking is based on both semantic relevance and lexical relevance
- [ ] Section-aware and metadata-aware retrieval is available

## 2. Hybrid search and reranking

- [ ] Dense retrieval is paired with keyword/lexical matching for exact terms
- [ ] Queries containing IDs, names, code values, error codes, and acronyms use lexical support
- [ ] Reranking is used to improve second-pass precision
- [ ] A larger candidate set is retrieved before reranking
- [ ] Top results are re-scored with relevance-aware logic
- [ ] Retrieval quality is measured before and after reranking

## 3. Embeddings and indexing

- [ ] The same embedding model is used consistently at ingest and query time
- [ ] Embedding model choice matches the domain and cost requirements
- [ ] Vector store is durable and version-aware
- [ ] Chunks retain useful metadata: source, section, page, version, document ID
- [ ] Index rebuilds are deterministic and repeatable
- [ ] Re-indexing handles changed documents without stale data
- [ ] Old versions are cleaned up or deactivated correctly

## 4. Document lifecycle and freshness

- [ ] Documents have unique IDs and version tracking
- [ ] New uploads create a new version, not silent overwrite
- [ ] Active document state is tracked properly
- [ ] Re-index jobs are async, observable, and retryable
- [ ] Document replacements deactivate old collections or versions cleanly
- [ ] Indexes can be rebuilt from source-of-truth data
- [ ] A document change triggers reprocessing when needed

## 5. Query handling and rewrite

- [ ] Query rewriting handles synonyms and domain language
- [ ] Common paraphrases are normalized before retrieval
- [ ] Acronyms and domain phrases do not break retrieval quality
- [ ] History-aware question flow does not inject stale context incorrectly
- [ ] Multi-turn queries preserve conversation context without leaking unrelated data

## 6. Grounding and refusal behavior

- [ ] The system refuses or abstains when evidence is weak
- [ ] The model is instructed to answer only from retrieved context
- [ ] Answers indicate uncertainty when evidence is insufficient
- [ ] Retrieval quality gates generation to prevent hallucinated replies
- [ ] Prompt construction includes only relevant evidence, not the full document
- [ ] Citations or source references are traceable back to retrieved chunks

## 7. Evaluation and benchmarking

- [ ] There is a benchmark set of real user questions
- [ ] Retrieval accuracy is measured separately from answer quality
- [ ] Answer groundedness is measured against retrieved evidence
- [ ] Hallucination rate is tracked over time
- [ ] Top-k retrieval hit rate is reviewed before deployment
- [ ] Regression tests cover chunking, retrieval, and reranking changes
- [ ] Evaluation is run before shipping a retrieval change

## 8. Observability and debugging

- [ ] Each request logs the query, top retrieved chunks, and scores
- [ ] Retrieval diagnostics are available in debug mode for troubleshooting
- [ ] Latency is measured per retrieval step
- [ ] Token usage and cost are tracked per request
- [ ] Failed or weak retrieval cases are visible in monitoring
- [ ] You can answer: “why was this chunk retrieved?”
- [ ] You can answer: “why did the model answer this way?”

## 9. Security and governance

- [ ] Access control is enforced per user, tenant, or document scope
- [ ] One tenant cannot query another tenant’s documents
- [ ] PII is filtered or protected before indexing and generation
- [ ] Sensitive documents are not exposed through broad retrieval
- [ ] Uploads are validated for type and size
- [ ] API auth is enforced for document ingestion and query endpoints
- [ ] Logs do not expose private context or sensitive content

## 10. Scalability and operations

- [ ] Indexing runs asynchronously for large documents
- [ ] Job status and retries are managed properly
- [ ] Background workers can recover from transient failures
- [ ] Re-index operations are idempotent when possible
- [ ] Storage and compute costs are understood and monitored
- [ ] Horizontal scaling strategy exists for multiple workers or instances
- [ ] Backups and restore paths exist for the vector store and metadata

## 11. Production-ready UX expectations

- [ ] The app shows progress for long-running indexing jobs
- [ ] Users can see whether a document is active or old
- [ ] Query results include traceable evidence
- [ ] System behavior is explainable to users and engineers
- [ ] Errors are user-friendly and do not leak raw internals
- [ ] Search can answer “no evidence found” cleanly when appropriate

## 12. Recommended maturity bar before calling it production

A system is usually ready for production when all of the following are true:

- [ ] Retrieval quality is measurable and improving
- [ ] Document lifecycle and re-indexing are safe and repeatable
- [ ] Reranking and hybrid search are part of the pipeline
- [ ] There is a clear refusal strategy for weak evidence
- [ ] The system can be debugged with chunk-level evidence
- [ ] There is job-level and request-level observability
- [ ] Access control and governance are enforced
- [ ] There is a real evaluation harness for regressions

## 13. The next step after this checklist

For a real RAG product, the next layer is usually:

1. stronger evaluation dataset
2. hybrid retrieval + reranking tuning
3. agentic orchestration for multi-step research
4. retrieval from multiple sources and tools
5. cost and latency monitoring
6. multi-tenant and permission-aware retrieval

This is the point where a RAG app becomes a real knowledge system rather than a single-prompt demo.

## 14. A short production principle

> Good RAG is not just a good model. It is a well-managed system of retrieval, ranking, validation, lifecycle control, and observability.
