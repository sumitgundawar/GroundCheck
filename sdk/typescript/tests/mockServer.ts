import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export interface Step {
  status?: number;
  /** An object is sent as JSON; a string or Buffer is sent as is. */
  body?: unknown;
  headers?: Record<string, string>;
  delayMs?: number;
  drop?: boolean;
  trickle?: boolean;
}

export interface Seen {
  method: string;
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

export class MockServer {
  requests: Seen[] = [];
  steps: Step[] = [];
  fallback: Step = { status: 200, body: {} };
  url = "";
  private server!: Server;

  async start(): Promise<this> {
    this.server = createServer((req, res) => void this.handle(req, res));
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
    return this;
  }

  async stop(): Promise<void> {
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }

  enqueue(...steps: Step[]): void {
    this.steps.push(...steps);
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    this.requests.push({
      method: req.method ?? "",
      path: req.url ?? "",
      headers: req.headers,
      body: Buffer.concat(chunks).toString("utf8"),
    });
    const step = this.steps.shift() ?? this.fallback;
    if (step.delayMs) await new Promise((r) => setTimeout(r, step.delayMs));
    if (step.drop) {
      req.socket.destroy();
      return;
    }
    const raw =
      typeof step.body === "string" || Buffer.isBuffer(step.body)
        ? (step.body as string | Buffer)
        : JSON.stringify(step.body ?? {});
    res.writeHead(step.status ?? 200, {
      "Content-Type": "application/json",
      ...(step.trickle ? { "Content-Length": "1000" } : {}),
      ...step.headers,
    });
    if (step.trickle) {
      for (let i = 0; i < 1000 && !res.destroyed; i++) {
        res.write(" ");
        await new Promise((r) => setTimeout(r, 50));
      }
      return;
    }
    res.end(raw);
  }
}
