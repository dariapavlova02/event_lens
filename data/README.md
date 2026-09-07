# Local data

Raw Telegram messages, API batch payloads, embeddings, model checkpoints and generated datasets are not distributed in this repository. Keep existing local data here; `.gitignore` excludes it from Git.

The current experiment needs both `btc_messages_tbsa.csv` and `all_messages_tbsa.csv`, plus complete hourly market observations. The latter message file historically excludes the BTC subset; it is not a complete corpus by itself. A fresh clone does not contain these inputs and is not an end-to-end reproducible benchmark yet.

Historical extraction scripts accept the local export path through `TELEGRAM_EXPORT_DIR`. Scripts that read environment variables directly need them exported in the shell; copying `.env.example` alone does not load them. Do not run API or Neo4j scripts without reviewing their side effects.
