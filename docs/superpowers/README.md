# Superpowers docs — specs and implementation plans

**First-time readers:** start with the repo guide [Onboarding](../ONBOARDING.md) (reading order for all docs + code + tests). System map (libraries ↔ tasks): [Architecture](../ARCHITECTURE.md). For failures and odd behavior, see [Debugging](../DEBUGGING.md).

Design (**what / why**) lives under `specs/`; step-by-step delivery notes (**where code lives**) under `plans/`. Each shipped phase doc includes **## What this phase uses** (stack, modules, config). Paths below are relative to this folder.

## Specs (`specs/`)

| Document | Topic |
|----------|--------|
| [`specs/2026-04-16-multimodal-rag-phase1-6-design.md`](./specs/2026-04-16-multimodal-rag-phase1-6-design.md) | Roadmap Phases 1–6, module boundaries |
| [`specs/2026-04-17-phase-3-hybrid-rerank-design.md`](./specs/2026-04-17-phase-3-hybrid-rerank-design.md) | Dense + BM25 + rerank |
| [`specs/2026-04-17-phase-4-cache-multihop-design.md`](./specs/2026-04-17-phase-4-cache-multihop-design.md) | Semantic cache, multi-hop; links query-refinement extension |
| [`specs/2026-04-17-phase-4-query-refinement-design.md`](./specs/2026-04-17-phase-4-query-refinement-design.md) | Optional LLM rewrite + dual retrieval merge |
| [`specs/2026-04-17-phase-5-tables-images-design.md`](./specs/2026-04-17-phase-5-tables-images-design.md) | Tables, image captions, modality router |
| [`specs/2026-04-17-phase-6-visual-retrieval-design.md`](./specs/2026-04-17-phase-6-visual-retrieval-design.md) | CLIP visual index + fusion; ColPali note |

## Plans (`plans/`)

| Document | Topic |
|----------|--------|
| [`plans/2026-04-16-phase-1-text-rag-implementation.md`](./plans/2026-04-16-phase-1-text-rag-implementation.md) | Phase 1 baseline |
| [`plans/2026-04-17-phase-2-evaluation-implementation.md`](./plans/2026-04-17-phase-2-evaluation-implementation.md) | Eval harness |
| [`plans/2026-04-17-phase-3-hybrid-rerank-implementation.md`](./plans/2026-04-17-phase-3-hybrid-rerank-implementation.md) | Hybrid + rerank wiring |
| [`plans/2026-04-17-phase-4-cache-multihop-implementation.md`](./plans/2026-04-17-phase-4-cache-multihop-implementation.md) | Cache + multihop; extension appendix |
| [`plans/2026-04-17-phase-4-query-refinement-implementation.md`](./plans/2026-04-17-phase-4-query-refinement-implementation.md) | **Code map** for dual-query refinement |
| [`plans/2026-04-17-phase-5-tables-images-implementation.md`](./plans/2026-04-17-phase-5-tables-images-implementation.md) | Tables / images / router |
| [`plans/2026-04-17-phase-6-visual-retrieval-implementation.md`](./plans/2026-04-17-phase-6-visual-retrieval-implementation.md) | CLIP + ColPali file map |

## Primary code entrypoints (quick reference)

- **Orchestration:** `main.py` — `index_pdf`, `run_rag_query`, `answer_query`
- **Retrieval:** `retrieval/pipeline.py`, `retrieval/multihop.py`, `retrieval/dual_query_merge.py`, `retrieval/colpali_retrieval.py`, `retrieval/visual_embedder.py`, `retrieval/visual_fusion.py`
- **Generation:** `generation/prompt_builder.py`, `generation/query_refinement.py`, `generation/llm_pipeline.py`, `generation/multihop_prompts.py`
- **Cache:** `cache/semantic_cache.py`, `cache/index_fingerprint.py`
- **UI:** `ui/app.py`, `ui/context_evidence.py`
- **Config:** `configs/settings.py`, `.env.example`

Repo root **README** summarizes runbooks and points here for deep dives.
