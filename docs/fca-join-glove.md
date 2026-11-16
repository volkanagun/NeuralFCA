# Pair-conditioned FCA concept selection for GloVe

Resnik (1995) measures similarity in an IS-A taxonomy using the information content of the most informative shared subsumer: [Using Information Content to Evaluate Semantic Similarity in a Taxonomy](https://www.ijcai.org/Proceedings/95-1/Papers/059.pdf).

**Our FCA adaptation** scores candidate nodes `a` and `b` using the information content of their join `a ∨ b`: the most specific shared superconcept under extent inclusion. We estimate structural information content from the lattice's object support:

`IC(c) = log((N + 1) / (|extent(c)| + 1))`, where `N` is the number of lattice objects.

The join is used only for selection. The GloVe vectors remain `[word_i; concept_a]` and `[context_word_j; concept_b]`, so each lexical type shares one trainable word vector and each lattice node has one trainable concept vector. With 300-dimensional blocks, each side is 600-dimensional.

The selector has two modes:

- **Word-pair mode** (`contextConditioned = false`) selects the highest-scoring `(a, b)` using `IC(a ∨ b)`. Every occurrence of the same ordered word pair gets the same concept rows.
- **Occurrence-aware mode** (`contextConditioned = true`) also scores each candidate against concept candidates of other tokens within the context radius. The target and context sides use their own nearby tokens, excluding the paired token. This can select different rows for two occurrences of the same word pair.

Selection happens before counting. Training aggregates the reciprocal-distance co-occurrence weights by `(word_i, a, word_j, b)` and applies the usual weighted GloVe squared error to each aggregated count. If a word has no candidate concept, its concept block is zero. If the lattice graph contains no shared ancestor, the join information-content score is zero.

Each checkpoint records its selector mode and rejects loading under the other mode. Use separate checkpoint paths when comparing the two experiments.

`ConceptGloveTrainer calibrate [words...]` reports the selected-pair count distribution and the scarcest focus words without updating parameters. `train-focused words...` trains on sentences containing those words as exact tokens; the normal trainer uses its configured `targetWords` in the same way. The sample limit counts tokens in eligible sentences. Training also prints the count report before the first epoch.

The GloVe weight is `min(count / xMax, 1)^alpha`. If every selected count is below `xMax`, changing `xMax` rescales all weights equally; lowering `alpha` increases the relative weight of low-count pairs. Neither parameter creates an observation for a word absent from the selected corpus. The counts include reciprocal-distance contributions and can be fractional.

This join-based selection rule and its occurrence-aware extension are FCA adaptations, not claims about Resnik's model.
