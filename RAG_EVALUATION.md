# RAG evaluation: what to measure and why

RAG evaluation is a repeatable way to check both halves of a retrieval-augmented
generation system:

1. **Retrieval** must find the evidence that answers a question.
2. **Generation** must answer the question accurately, using that evidence and
   nothing unsupported.

An answer can sound convincing while either half fails. A reliable evaluation
therefore checks retrieved evidence, answer quality, and the time spent in each
stage rather than relying on a few manual chat examples.

## Metrics in this app

| Metric | What it tells you | How this app measures it |
|---|---|---|
| **Recall@k** | Was the expected evidence retrieved in the top *k* chunks? | For each question, label one or more relevant locations as `filename:page` (PDF) or `filename:row` (CSV). Recall is the fraction of those labeled locations present in the retrieved chunks. |
| **Relevance** | Does the answer address the question and agree with the expected answer? | An LLM judge compares the question, your reference answer, and the generated answer. It returns a score from 0 (not relevant/correct) to 1 (fully relevant/correct) and a short reason. |
| **Groundedness** | Are the answer's factual claims supported by the retrieved evidence? | The same judge compares the answer only with the retrieved context. It returns a score from 0 (unsupported) to 1 (fully supported) and a reason. |
| **Latency per stage** | Where is query time being spent? | The app measures retrieval and answer generation in milliseconds. The evaluation table also reports judge time and total case time; judge time is evaluation overhead, not RAG answer latency. |

### Interpreting the scores

- **Low Recall@k** usually points to chunking, embeddings, query rewriting,
  ranking, or `k`. The generator cannot use a chunk it never received.
- **High Recall@k but low relevance** means the evidence was found, but the
  answer may be incomplete, incorrect, or poorly prompted.
- **High relevance but low groundedness** is a warning: the answer may match
  the expected answer while adding claims that the retrieved text does not
  support.
- **Low latency is not a quality metric.** Compare latency alongside the
  quality scores when changing the RAG implementation or index settings.

LLM-judge scores are estimates, not objective truth. Keep the same judge model
and rubric when comparing runs, review the explanations for surprising scores,
and use human review for high-impact decisions.

## Running an evaluation

1. In the app, upload and index the documents your questions refer to.
2. Open **RAG evaluation** and add one row per test question.
3. Fill in:
   - **Question**: the user question to run through the selected RAG mode.
   - **Reference answer**: the expected answer, used by the relevance judge.
   - **Gold locations**: comma-separated source locations, for example
     `employee-policy.pdf:2` or `employee-policy.pdf:2, employee-policy.pdf:5`.
     PDF locations use page numbers; CSV locations use row numbers.
4. Choose `k`, the distance threshold, and the RAG implementation in the
   sidebar, then click **Run evaluation**.
5. Compare the aggregate metrics and per-question explanations. Expand a case
   to inspect the answer and the retrieved chunks that were judged.

Gold location labels must use the exact filename shown in the indexed sources.
Use every source location that contains evidence needed for the answer; Recall@k
is calculated over the unique labels provided for that question. A test set
should include answerable and unanswerable questions if refusal behavior is
important, and should be maintained as application behavior or documents change.

For large CSVs, the sidebar's **CSV rows per embedding chunk** setting groups
consecutive rows to reduce embedding requests and vector-store writes. The
original row numbers remain in the chunk text and metadata, and Recall@k treats
a gold row as found when it falls inside a retrieved chunk's row range. Re-index
the CSV after changing this setting. Grouping can make each retrieved context
larger and may include neighboring rows that are irrelevant to a given question.
The character chunk-size setting also caps normal groups; a row longer than that
limit may still be split across multiple chunks.

## Example

Suppose `employee-policy.pdf` says on page 2 that staff receive 20 days of
annual leave. A test case could be:

```text
Question: How many annual leave days do staff receive?
Reference answer: Staff receive 20 days of annual leave.
Gold locations: employee-policy.pdf:2
```

If page 2 is among the top-*k* retrieved chunks, Recall@k is 1. If the answer
correctly says 20 days and the claim is supported by the retrieved context, the
judge should score relevance and groundedness near 1. Retrieval and generation
latencies show how long those two RAG stages took for this query.

## Limits and next steps

This learning app uses labeled page/row locations as a practical proxy for
relevant chunks. It does not yet measure precision, ranking quality such as
MRR/nDCG, citation correctness, refusal quality, or confidence intervals. The
LLM judge adds an API call per evaluation case, so account for its time and
model cost separately from the RAG stage timings.
