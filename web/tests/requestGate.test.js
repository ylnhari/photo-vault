import assert from "node:assert/strict";
import test from "node:test";
import { createRequestGate } from "../src/lib/requestGate.js";

test("starting a newer request aborts and invalidates the previous one", () => {
  const gate = createRequestGate();
  const older = gate.begin();
  const newer = gate.begin();

  assert.equal(older.signal.aborted, true);
  assert.equal(older.isCurrent(), false);
  assert.equal(newer.signal.aborted, false);
  assert.equal(newer.isCurrent(), true);
});

test("cancel invalidates a request even if its provider ignores abort", () => {
  const gate = createRequestGate();
  const request = gate.begin();

  gate.cancel();

  assert.equal(request.signal.aborted, true);
  assert.equal(request.isCurrent(), false);
});
