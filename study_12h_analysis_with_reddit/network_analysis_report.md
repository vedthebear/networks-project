# Moltbook and Reddit Network Analysis

Generated: `2026-05-28T21:00:18.754786+00:00`

## Moltbook Actor Graph

- **Nodes:** 1476
- **Directed edges:** 4837
- **Directed density:** 0.00222176
- **Largest weak component share:** 0.947154
- **Largest strong component share:** 0.340786
- **Reciprocity:** 0.393219
- **Weighted reciprocity:** 0.350518
- **Average local clustering:** 0.270589
- **Weighted in-degree Gini:** 0.848725
- **Top 1% inbound share:** 0.30828
- **Top 5% inbound share:** 0.623634

Graph variants:

- **Comment-on-post nodes:** 1467
- **Comment-on-post edges:** 3716
- **Reply-to-comment nodes:** 554
- **Reply-to-comment edges:** 1258
- **Uncapped/fully observed nodes:** 1233
- **Uncapped/fully observed edges:** 4497

Top PageRank agents:

- `c7a8289f-3eb5-42a2-8a62-8e9ca69e734b`: 0.0982423
- `0e118b5f-7987-40e8-bbc8-db7a39d30b32`: 0.0240247
- `d38937a9-3f0e-4bae-a53b-0136cdb244cd`: 0.0207212
- `87c61e20-d6a2-4bb7-b09a-902ec1152086`: 0.0200466
- `132a306f-f126-4e15-9075-9aa2f5762e80`: 0.0172732
- `d6936b86-9755-4317-9277-98ebb8e48808`: 0.0135389
- `574fe716-e6fe-418a-b21c-a1a6663c247d`: 0.012163
- `2fc823f4-bc45-4f1c-b61c-ac4719db5efe`: 0.0120953
- `d95195ec-c40c-45b5-b711-97804518a950`: 0.0120953
- `dac940c5-a963-49f7-8f9e-6b1661051af3`: 0.0117045

## Moltbook Coverage

- **Posts with coverage rows:** 5754
- **Average coverage ratio:** 1.00005
- **Capped or under-observed posts:** 33
- **Unknown expected-comment posts:** 2407
- **Observed comments across coverage rows:** 14276
- **Expected comments across coverage rows:** 83435

## Reddit Thread Graphs

- **Dataset:** reddit_threads
- **Thread count:** 0
- **Average nodes/thread:** None
- **Average edges/thread:** None
- **Average density:** None

## Reddit Temporal Actor Graph

- **Nodes:** 0
- **Directed edges:** 0
- **Reciprocity:** 0
- **Weighted in-degree Gini:** 0

## Reddit SNAP Reply/Chain Graphs

- **Dataset:** snap_reddit
- **Reply graph skipped:** True
- **Reply graph nodes:** None
- **Reply graph directed edges:** None
- **Reply graph self-loop edges:** None
- **Chain graph skipped:** True
- **Chain graph nodes:** None
- **Chain graph directed edges:** None
- **Chain graph self-loop edges:** None
- **Chain weighted in-degree Gini:** None

## Interpretation Notes

- Compare Moltbook actor metrics to Reddit temporal actor metrics when using a user-reply edge stream.
- Compare Moltbook post/thread shapes to Reddit thread graphs when using Reddit Threads.
- Treat coverage metrics as a warning light: low coverage means comment caps or endpoint limits still need deeper hydration.
- For a fair final paper, compare matched time windows and matched sample sizes, then bootstrap results.
