import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Snapshot } from "@/lib/tactics";

/**
 * The agent loop itself, driven by a scripted model.
 *
 * These tests replace the Anthropic client with one that returns a fixed
 * sequence of responses, so the loop's mechanics can be checked without a
 * network call or an API key. What they pin is the plumbing that is easy to get
 * wrong and silent when it is: that tool results actually get executed and fed
 * back, that several tool calls in one turn come back in a single message, and
 * that the loop cannot run forever.
 */

const create = vi.fn();

vi.mock("@anthropic-ai/sdk", () => ({
  default: class {
    messages = { create };
  },
}));

const { investigate, MAX_TURNS } = await import("./route");

function snapshot(frame: number): Snapshot {
  return {
    frame,
    timeS: frame / 25,
    ball: { x: -20, y: 0 },
    players: [
      { id: 1, team: "team_a", x: -20, y: 0, label: "1" },
      { id: 2, team: "team_a", x: 20, y: 0, label: "2" },
      { id: 10, team: "team_b", x: 30, y: -10, label: "10" },
      { id: 11, team: "team_b", x: 30, y: 10, label: "11" },
    ],
  };
}

const clip = [0, 1, 2].map(snapshot);

const REPORT = JSON.stringify({
  headline: "They played through the middle.",
  findings: [{ claim: "c", evidence: "e", frames: [1] }],
  limitations: "short clip",
});

function toolTurn(calls: Array<{ id: string; name: string; input: unknown }>) {
  return {
    stop_reason: "tool_use",
    content: calls.map((c) => ({ type: "tool_use", ...c })),
  };
}

function finalTurn(text = REPORT) {
  return { stop_reason: "end_turn", content: [{ type: "text", text }] };
}

beforeEach(() => {
  create.mockReset();
});

describe("investigate", () => {
  it("executes a requested tool and feeds the result back", async () => {
    create
      .mockResolvedValueOnce(
        toolTurn([{ id: "t1", name: "clip_summary", input: {} }]),
      )
      .mockResolvedValueOnce(finalTurn());

    const result = await investigate("why?", clip, "key");

    expect(result.turnsUsed).toBe(2);
    expect(result.toolCalls).toBe(1);

    // The second request must carry the tool result back, or the model is
    // answering from nothing and the whole loop is theatre.
    const secondRequest = create.mock.calls[1][0];
    const lastMessage = secondRequest.messages[secondRequest.messages.length - 1];
    expect(lastMessage.role).toBe("user");
    expect(lastMessage.content[0].type).toBe("tool_result");
    expect(lastMessage.content[0].tool_use_id).toBe("t1");
  });

  it("returns several tool results in one message, not one message each", async () => {
    /*
     * Splitting parallel tool results across messages is accepted by the API
     * and quietly trains the model to stop requesting tools in parallel. It
     * would never surface as an error, only as a slower agent.
     */
    create
      .mockResolvedValueOnce(
        toolTurn([
          { id: "a", name: "clip_summary", input: {} },
          { id: "b", name: "inspect_frame", input: { frame: 1 } },
        ]),
      )
      .mockResolvedValueOnce(finalTurn());

    await investigate("why?", clip, "key");

    const secondRequest = create.mock.calls[1][0];
    const lastMessage = secondRequest.messages[secondRequest.messages.length - 1];
    expect(lastMessage.content).toHaveLength(2);
    expect(lastMessage.content.map((c: { tool_use_id: string }) => c.tool_use_id)).toEqual([
      "a",
      "b",
    ]);
  });

  it("records every tool call in the transcript", async () => {
    // The transcript is the audit trail: it is what lets a reader check a
    // finding against the measurement it came from.
    create
      .mockResolvedValueOnce(
        toolTurn([{ id: "t1", name: "inspect_frame", input: { frame: 2 } }]),
      )
      .mockResolvedValueOnce(finalTurn());

    const result = await investigate("why?", clip, "key");

    expect(result.transcript).toHaveLength(1);
    expect(result.transcript[0].tool).toBe("inspect_frame");
    expect(result.transcript[0].input).toEqual({ frame: 2 });
    expect(result.transcript[0].result).toMatchObject({ frame: 2 });
  });

  it("withholds tools on the final turn so the loop cannot stall", async () => {
    /*
     * Without this the model can spend its last turn asking for a search it
     * will never receive, and the investigation ends with no report at all.
     */
    create.mockResolvedValue(
      toolTurn([{ id: "t", name: "clip_summary", input: {} }]),
    );

    await expect(investigate("why?", clip, "key")).rejects.toThrow(/did not conclude/);

    const lastRequest = create.mock.calls[create.mock.calls.length - 1][0];
    expect(lastRequest.tools).toBeUndefined();
    expect(create).toHaveBeenCalledTimes(MAX_TURNS);
  });

  it("surfaces a refusal rather than returning an empty report", async () => {
    create.mockResolvedValueOnce({ stop_reason: "refusal", content: [] });
    await expect(investigate("why?", clip, "key")).rejects.toThrow(/declined/);
  });

  it("passes an errored tool result back instead of throwing", async () => {
    /*
     * A bad frame number is the model's mistake to recover from, not a reason
     * to abort the investigation. It has to see the error to correct course.
     */
    create
      .mockResolvedValueOnce(
        toolTurn([{ id: "t1", name: "inspect_frame", input: { frame: 999 } }]),
      )
      .mockResolvedValueOnce(finalTurn());

    const result = await investigate("why?", clip, "key");

    expect(result.turnsUsed).toBe(2);
    expect(JSON.stringify(result.transcript[0].result)).toContain("999");
  });

  it("asks for structured output on every turn", async () => {
    create.mockResolvedValueOnce(finalTurn());
    await investigate("why?", clip, "key");

    const request = create.mock.calls[0][0];
    expect(request.output_config.format.type).toBe("json_schema");
    expect(request.model).toBe("claude-opus-5");
  });
});
