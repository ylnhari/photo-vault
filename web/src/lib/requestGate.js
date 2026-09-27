// Abort the previous request and give each new request a safe state token.
// The token check also protects callers when a mocked or cached request ignores abort.
export function createRequestGate() {
  let generation = 0;
  let controller = null;

  return {
    begin() {
      controller?.abort();
      controller = new AbortController();
      const current = ++generation;
      const active = controller;
      return {
        signal: active.signal,
        isCurrent: () => current === generation && !active.signal.aborted,
      };
    },
    cancel() {
      generation += 1;
      controller?.abort();
      controller = null;
    },
  };
}
