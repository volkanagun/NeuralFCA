# Sentence-based word and FCA concept training

Run `llm.LatticeGraveTrainer` with the `sentence` argument to train and export the new model. Run it with `sentence-export` to export an existing sentence-model checkpoint. The original `train` and `export` arguments still run the GloVe baseline.

`SentenceConceptTrainer` streams the configured sentence file without selecting lines by evaluation vocabulary. Each known center word is hidden in a fixed, bidirectional window. The model predicts its word ID using sampled softmax over the trainable `wordEmb` table. It also predicts a context-selected FCA concept using sampled softmax over `conceptEmb`; other concepts of the target word are included as hard alternatives. The positive concept comes from the lattice join selector and visible neighboring words. The center word and its concept are absent from the encoder input, so neither prediction can read the answer directly.

The checkpoint uses separate paths from GloVe and records a row mapping plus a sentence-model configuration sidecar. Export writes `vectors-sentence-concept.txt` and `simple-sentence-concept-lattice.bin` through `AttentionLatticeUpdater`.

The small CPU round-trip test is `testing.SentenceConceptTrainerTest`. Compare held-out word similarity and sentence-specific concept selection against GloVe before claiming an embedding-quality improvement on the full Turkish corpus.
