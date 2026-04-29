# Cache

The two cache classes you instantiate. Both share a Protocol-shaped surface; the async one is a thin wrapper that awaits the embedder directly and dispatches blocking store work to a thread pool. Behavior is otherwise identical.

::: mneme._cache.SemanticCache

::: mneme._async_cache.AsyncSemanticCache
