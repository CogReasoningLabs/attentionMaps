# Modularization architecture and guardrails

## Outcome

The two application modules that exceeded 1,000 lines have been split without
changing their Streamlit entry commands:

| Previous hotspot | Previous size | New composition root | Extracted modules |
| --- | ---: | ---: | --- |
| `apps/dataset_explorer.py` | 5,044 | 681 lines | dataset catalog, inspection, text/presentation services, shared components, and feature-oriented tab renderers |
| `apps/dataset_generator.py` | 1,119 | 876 lines | generation family registry, record/schema services, shared components, and SQLite persistence |

Every Python module in `apps/`, `attention_maps/`, and `scripts/` is now at or
below the 1,000-line hard limit. `tests/test_module_size_budget.py` enforces
that limit in CI. The preferred review target is much smaller: one cohesive
responsibility per module, usually below 500 lines. Line count is a guardrail,
not a substitute for cohesion.

## Research-backed principles

- Python's official tutorial recommends modules for splitting long programs
  and packages for structuring a module namespace. The explorer therefore has
  a reusable `attention_maps.explorer` package instead of keeping domain logic
  in a Streamlit script: <https://docs.python.org/3/tutorial/modules.html>.
- PEP 8 recommends absolute imports and explicit public/internal interfaces.
  New package APIs use explicit imports and `__all__`; private generator
  compatibility names remain private: <https://peps.python.org/pep-0008/>.
- Streamlit recommends defining custom classes in separate modules so reruns do
  not redefine them. `DatasetSpec` now lives in the catalog module:
  <https://docs.streamlit.io/develop/concepts/design/custom-classes>.
- Streamlit distinguishes serializable data caching (`st.cache_data`) from
  shared resources such as models (`st.cache_resource`). The composition roots
  retain that split while renderers receive cached loaders as dependencies:
  <https://docs.streamlit.io/develop/concepts/architecture/caching>.
- Streamlit's page model supports independently executable UI units. The
  current tabs remain stable for users, but each feature is now a page-sized
  renderer that can later move behind `st.Page` navigation without moving its
  domain code again: <https://docs.streamlit.io/develop/api-reference/navigation/st.page>.

## Dependency direction

```text
apps/dataset_explorer.py (composition, cache wiring, navigation)
        │
        ├── apps/components.synthetic_data
        │            └── attention_maps.generation.catalog
        │                         └── attention_maps.explorer.catalog
        │
        ├── apps/explorer_tabs/* (feature renderers)
        │            │
        │            └── attention_maps.explorer + inference/evaluation APIs
        │
        └── attention_maps.explorer
                 ├── catalog.py       taxonomy and DatasetSpec
                 ├── inspection.py    bounded source inspection/sampling
                 ├── text.py          Unicode/text analysis helpers
                 └── presentation.py  record/manifest presentation helpers

apps/dataset_generator.py (composition, cache wiring, generation UI)
        ├── apps/components.synthetic_data
        ├── attention_maps.generation.catalog
        ├── attention_maps.generation.records
        ├── attention_maps.generation.persistence
        └── attention_maps.explorer + inference APIs
```

Domain packages do not import Streamlit or either app entrypoint. UI renderers
receive session-specific values and cached loader functions explicitly. This
keeps the heavy model/data cache lifecycle in one place and makes domain code
usable from tests, scripts, and future pages.

## Change rules

1. Put catalog, sampling, Unicode, persistence, scoring, and model-loading
   behavior under `attention_maps/`; keep `apps/` focused on UI composition.
2. Add a renderer to the feature module that owns the tab. Do not create
   cross-tab variables; return a value or pass it explicitly.
3. Cache bounded/serializable query results with `st.cache_data`; cache model
   clients and loaded model/tokenizer resources with `st.cache_resource`.
4. Preserve expensive imports inside the feature or loader that needs them.
5. Add focused unit tests for extracted domain behavior and Streamlit
   characterization tests for dropdowns, navigation, and critical workflows.
6. If a module approaches 1,000 lines, split by responsibility before the
   budget test fails. Avoid arbitrary `utils.py` dumping grounds.
7. Register a new synthetic dataset family in
   `attention_maps/generation/catalog.py`; do not add family-specific selection
   branches independently to the explorer and generator.

## Verification

Run the architecture guard and full suite in the project environment:

```bash
.venv/bin/python -m unittest tests.test_module_size_budget -v
MPLCONFIGDIR=/tmp/matplotlib-cache \
  .venv/bin/python -m unittest discover -s tests -v
```
