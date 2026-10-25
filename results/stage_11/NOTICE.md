# Stage 11 attribution and execution boundary

Candidate source: Vladislav Lavrentev (lightningv08), exact Kaggle V28 **362014737**, displayed official public score **0.642**. Notebook source is released under Apache-2.0; the original upstream credits and individual model/dataset licences remain applicable.

The Stage 11 CPU checker is independently written preparation code. Its embedded contract contains source-derived factual paths, sizes, shapes and metadata. It does not import or execute the upstream source, custom model code, joblib/pickle payloads or Torch checkpoints. Input-code review exports remain private. No original competition passages, row predictions or trained weights are published here.

Our completed S04 scored **0.639 official public balanced accuracy**. No Stage 11 score is implied by source-author results, private preparation, CPU audits or runtime probes. The source's three-example shortcut is format-only and does not load the complete models. A GPU timing probe uses training texts for speed/numerical diagnostics only; the final adapter saw all official training labels, so no probe accuracy is reported as validation.
