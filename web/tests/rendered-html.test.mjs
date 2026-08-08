import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders fleet dashboard metadata and loading state", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>C-MAPSS Fleet Intelligence<\/title>/i);
  assert.match(html, /Predictive maintenance intelligence for NASA turbofan engine fleets/);
  assert.match(html, /Loading fleet intelligence/);
  assert.doesNotMatch(html, /Starter Project|Your site is taking shape/);
});

test("ships a complete verified dashboard data contract", async () => {
  const payload = JSON.parse(
    await readFile(new URL("../public/data/dashboard.json", import.meta.url), "utf8"),
  );

  assert.equal(payload.meta.dataset, "NASA C-MAPSS FD001");
  assert.equal(payload.engines.length, 100);
  assert.equal(payload.metrics.test_maintenance_value.expected_value, 6_200_000);
  assert.ok(payload.featureImportance.length > 0);
  assert.ok(payload.engines.every((engine) => engine.history.length > 0));
});

test("dashboard source includes primary operator interactions", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /Maintenance threshold/);
  assert.match(page, /Search engines/);
  assert.match(page, /setScheduled\(true\)/);
  assert.match(page, /type="range"/);
});
