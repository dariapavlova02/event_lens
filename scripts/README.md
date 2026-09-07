# Historical research scripts

Retained as implementation references for the next experiment and evidence for the audit. Their scientific and operational defects are not all fixed. Numbering reflects the historical experiments, not a recommended execution order.

Publication cleanup removes embedded database passwords and makes the main project data/results paths relative to this checkout. Set `NEO4J_PASSWORD` explicitly for database scripts and `TELEGRAM_EXPORT_DIR` for external source exports. Algorithmic results have not been revalidated after cleanup.

Read before executing: some scripts incur API costs, clear a Neo4j database or overwrite datasets. The next benchmark should reuse only the relevant parts and has not been implemented yet.
