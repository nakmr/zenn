---
title: "LangGraph Main Changes 2025-08-26"
emoji: "📝"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [LangGraph]
published: true
published_at: 2025-08-29 08:45
---

## [release(cli): 0.4.0 (#6014)](https://github.com/langchain-ai/langgraph/commit/ddf4e62bde806111a05de247cf028ec08883eac0)

- CLIのバージョンが 0.4.0 に上がりました
- `langgraph-api` と `langgraph-runtime-inmem`[^1] の参照バージョンが上がりました

[^1]: LangGraph APIサーバのランタイムを、Dockerを使わず、メモリのみで実行するための実装。CLIで `dev` コマンド実行時に指定できる ([Doc](https://docs.langchain.com/langgraph-platform/cli?utm_source=chatgpt.com#dev))
