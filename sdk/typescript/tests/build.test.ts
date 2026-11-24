import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const root = new URL("..", import.meta.url).pathname;

describe("the published package", () => {
  it("builds as ESM and CJS with types, both importable, with no runtime dependencies", () => {
    execFileSync("npm", ["run", "build", "--silent"], { cwd: root, stdio: "pipe" });
    for (const f of ["dist/esm/index.js", "dist/esm/index.d.ts", "dist/cjs/index.js", "dist/cjs/index.d.ts"]) {
      expect(existsSync(root + f), f).toBe(true);
    }
    const esm = execFileSync("node", ["--input-type=module", "-e",
      'import { GroundCheck, RateLimitError } from "@groundcheck/client"; console.log(typeof GroundCheck, new RateLimitError("x").name)'],
      { cwd: root }).toString().trim();
    const cjs = execFileSync("node", ["-e",
      'const m = require("@groundcheck/client"); console.log(typeof m.GroundCheck, new m.RateLimitError("x").name)'],
      { cwd: root }).toString().trim();
    expect(esm).toBe("function RateLimitError");
    expect(cjs).toBe("function RateLimitError");
    const pkg = JSON.parse(readFileSync(root + "package.json", "utf8"));
    expect(pkg.dependencies).toEqual({});
  }, 120_000);
});
