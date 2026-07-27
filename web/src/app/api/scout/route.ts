/**
 * The scout: a multi-step agent that answers questions about a whole clip.
 *
 * `/api/analyse` narrates one frame and `/api/search` translates one question
 * into one query. Neither can answer "how did they create their chances?",
 * because that needs several searches, a look at what each one returned, and a
 * decision about what to look at next. That is an agent loop, so this is one.
 *
 * The invariant is unchanged and is the reason this is safe to build. Every
 * tool here is a call into `lib/tactics`, which computes geometry from tracking
 * data. The model chooses *which* questions to ask and in what order; it never
 * computes a number, never decides whether a frame qualifies, and never sees a
 * coordinate. So the loop can be as agentic as it likes and the findings stay
 * as trustworthy as the engine underneath them.
 *
 * A manual loop rather than the SDK's tool runner, for two reasons: the runner
 * is beta, and every tool call here has to be checked against a whitelist and
 * counted before it executes. Owning the loop makes both explicit rather than
 * arranging them around a helper.
 */

import type AnthropicSDK from "@anthropic-ai/sdk";
import { NextRequest } from "next/server";

import type { Snapshot } from "@/lib/tactics";
import { analyseSnapshot } from "@/lib/tactics";
import {
  describeQuery,
  groupIntoSequences,
  searchFrames,
  type TacticalQuery,
} from "@/lib/tactics/search";

interface ScoutRequest {
  question?: string;
  snapshots?: Snapshot[];
  fps?: number;
}

/**
 * How many model turns the loop will take before it has to answer.
 *
 * Not a safety valve, a cost ceiling. Each turn is a full request carrying the
 * accumulated transcript, so an agent left to wander is the expensive failure
 * mode here, not the dangerous one. Six is enough for a search, a look, a
 * refined search, and a second look, which is the shape most questions take.
 */
const MAX_TURNS = 6;

/** Sequences returned per search, so one broad query cannot fill the context. */
const MAX_HITS_PER_SEARCH = 6;

const SYSTEM_PROMPT = `You are a football analyst studying tracking data from one clip.

You cannot see the pitch. You investigate by calling tools, which compute measurements from the tracking data. Everything you assert must come from a tool result.

Method:
1. Search for passages matching a tactical pattern.
2. Inspect specific frames the search returned to see what was actually happening.
3. Refine and search again if the first pattern was not the right one.
4. Report what you found.

Rules:
- Never state a number a tool did not give you. If you did not measure it, do not claim it.
- If the clip contains nothing matching the question, say so plainly. An honest empty answer is worth more than an invented pattern.
- A single frame is not a pattern. Two or three passages showing the same thing is.
- Cite the frame or time range behind every finding so it can be checked.

Write no em dashes, en dashes, or " - " as sentence punctuation. Use a comma, "and", or two sentences.`;

const TOOLS = [
  {
    name: "search_moments",
    description:
      "Find passages in the clip matching a tactical pattern. Every filter is " +
      "a threshold on a measured quantity and they combine with AND. Returns " +
      "time ranges with the measurements that qualified each one. Start broad: " +
      "each filter you add narrows the result, so over-specifying returns nothing.",
    input_schema: {
      type: "object",
      properties: {
        minOpenLanes: {
          type: "integer",
          description: "At least this many passing lanes judged open.",
        },
        minBestMarginS: {
          type: "number",
          description: "Best lane must beat the nearest defender by this many seconds.",
        },
        minProgressionM: {
          type: "number",
          description: "Lanes must gain at least this many metres toward goal.",
        },
        minBlockWidthM: { type: "number", description: "Defensive block at least this wide." },
        maxBlockWidthM: { type: "number", description: "Defensive block no wider than this." },
        minBlockDepthM: { type: "number", description: "Block at least this deep." },
        maxBlockDepthM: { type: "number", description: "Block no deeper than this." },
        minBackLineGapM: {
          type: "number",
          description: "Largest gap in the last defensive line, in metres. Over 12 is large.",
        },
        minPlayersBetweenLines: {
          type: "integer",
          description: "Attackers positioned between the opponent's lines.",
        },
        minPlayersBeyondLine: {
          type: "integer",
          description: "Attackers beyond the last defensive line.",
        },
        attackingTeam: {
          type: "string",
          enum: ["team_a", "team_b"],
          description: "Restrict to frames where this team has the ball.",
        },
      },
      additionalProperties: false,
      required: [],
    },
  },
  {
    name: "inspect_frame",
    description:
      "Full tactical report for one frame: every passing lane with its margin " +
      "and the defenders it bypasses, the defensive block's shape, and which " +
      "players sit between or beyond the lines. Use this on frames a search " +
      "returned, to find out what was actually happening in them.",
    input_schema: {
      type: "object",
      properties: {
        frame: { type: "integer", description: "Frame number, from a search result." },
      },
      additionalProperties: false,
      required: ["frame"],
    },
  },
  {
    name: "clip_summary",
    description:
      "Overview of the whole clip: length, which team had the ball and for how " +
      "long, and the range of defensive block shapes seen. Call this first to " +
      "orient yourself before searching.",
    input_schema: { type: "object", properties: {}, additionalProperties: false, required: [] },
  },
] as const;

const REPORT_SCHEMA = {
  type: "object",
  properties: {
    headline: {
      type: "string",
      description: "One sentence answering the question.",
    },
    findings: {
      type: "array",
      description: "Each finding must rest on a measurement a tool returned.",
      items: {
        type: "object",
        properties: {
          claim: { type: "string" },
          evidence: {
            type: "string",
            description: "The measurements and time ranges supporting this claim.",
          },
          frames: {
            type: "array",
            items: { type: "integer" },
            description: "Frames a reader can jump to in order to check it.",
          },
        },
        required: ["claim", "evidence", "frames"],
        additionalProperties: false,
      },
    },
    limitations: {
      type: "string",
      description:
        "What this analysis could not establish, including whether the clip was too short.",
    },
  },
  required: ["headline", "findings", "limitations"],
  additionalProperties: false,
} as const;

/** One tool call, executed against the deterministic engine. */
function runTool(
  name: string,
  input: Record<string, unknown>,
  snapshots: Snapshot[],
): unknown {
  switch (name) {
    case "search_moments": {
      const query = input as TacticalQuery;
      const sequences = groupIntoSequences(searchFrames(snapshots, query));
      return {
        interpretedAs: describeQuery(query),
        matchCount: sequences.length,
        passages: sequences.slice(0, MAX_HITS_PER_SEARCH).map((s) => ({
          startFrame: s.startFrame,
          endFrame: s.endFrame,
          startTimeS: Number(s.startTimeS.toFixed(1)),
          endTimeS: Number(s.endTimeS.toFixed(1)),
          frameCount: s.frameCount,
          peakFrame: s.peak.frame,
          measurements: s.peak.measurements,
        })),
      };
    }

    case "inspect_frame": {
      const frame = Number(input.frame);
      const index = snapshots.findIndex((s) => s.frame === frame);
      if (index < 0) {
        return { error: `frame ${frame} is not in this clip` };
      }
      const report = analyseSnapshot(snapshots[index], {
        previous: index > 0 ? snapshots[index - 1] : null,
        computeSpace: false,
      });
      if (!report) return { error: `frame ${frame} could not be analysed` };

      return {
        frame,
        timeS: Number(snapshots[index].timeS.toFixed(1)),
        attackingTeam: report.attackingTeam,
        carrier: report.carrierId,
        lanes: report.lanes.slice(0, 5).map((l) => ({
          target: l.targetId,
          verdict: l.verdict,
          marginS: Number.isFinite(l.safetyMarginS)
            ? Number(l.safetyMarginS.toFixed(2))
            : "unopposed",
          distanceM: Number(l.distanceM.toFixed(1)),
          progressionM: Number(l.progressionM.toFixed(1)),
          defendersBypassed: l.defendersBypassed,
        })),
        block: report.block
          ? {
              players: report.block.playerCount,
              widthM: Number(report.block.widthM.toFixed(1)),
              depthM: Number(report.block.blockDepthM.toFixed(1)),
              largestGapM: Number(report.block.largestBackLineGapM.toFixed(1)),
            }
          : null,
        playersBetweenLines: report.playersBetweenLines,
        playersBeyondLine: report.playersBeyondLine,
      };
    }

    case "clip_summary": {
      const possession: Record<string, number> = {};
      const widths: number[] = [];

      for (let i = 0; i < snapshots.length; i += 1) {
        const report = analyseSnapshot(snapshots[i], {
          previous: i > 0 ? snapshots[i - 1] : null,
          computeSpace: false,
        });
        if (!report) continue;
        possession[report.attackingTeam] = (possession[report.attackingTeam] ?? 0) + 1;
        if (report.block) widths.push(report.block.widthM);
      }

      widths.sort((a, b) => a - b);
      return {
        frames: snapshots.length,
        durationS: snapshots.length
          ? Number(snapshots[snapshots.length - 1].timeS.toFixed(1))
          : 0,
        framesInPossession: possession,
        blockWidthM: widths.length
          ? {
              min: Number(widths[0].toFixed(1)),
              median: Number(widths[Math.floor(widths.length / 2)].toFixed(1)),
              max: Number(widths[widths.length - 1].toFixed(1)),
            }
          : null,
      };
    }

    default:
      return { error: `no such tool: ${name}` };
  }
}

export async function POST(request: NextRequest) {
  let body: ScoutRequest;
  try {
    body = (await request.json()) as ScoutRequest;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const question = body.question?.trim();
  const snapshots = body.snapshots;
  if (!question) {
    return Response.json({ error: "provide `question`" }, { status: 400 });
  }
  if (!Array.isArray(snapshots) || !snapshots.length) {
    return Response.json({ error: "provide `snapshots`" }, { status: 400 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    // No deterministic substitute is offered here, unlike the other two routes.
    // Narration and query translation have honest rule-based fallbacks because
    // the work is phrasing and pattern matching. Deciding what to investigate
    // next based on what the last search returned is the part a model actually
    // does, and a canned sequence of searches pretending to be an investigation
    // would be worse than saying plainly that this needs a key.
    return Response.json(
      {
        error:
          "The scout runs a multi-step investigation and needs ANTHROPIC_API_KEY. " +
          "Frame analysis and clip search both work without one.",
      },
      { status: 503 },
    );
  }

  try {
    const result = await investigate(question, snapshots, apiKey);
    return Response.json(result);
  } catch (error) {
    console.error("scout investigation failed:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "investigation failed" },
      { status: 500 },
    );
  }
}

async function investigate(question: string, snapshots: Snapshot[], apiKey: string) {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ apiKey });

  const messages: AnthropicSDK.MessageParam[] = [
    { role: "user", content: `Question about this clip: ${question}` },
  ];

  // The transcript of what was actually measured, kept alongside the model's
  // messages. This is the audit trail: it says which tools ran, with what
  // arguments, and what came back, so a finding can be traced to a measurement
  // rather than taken on trust.
  const toolLog: Array<{ tool: string; input: unknown; result: unknown }> = [];

  for (let turn = 0; turn < MAX_TURNS; turn += 1) {
    const lastTurn = turn === MAX_TURNS - 1;

    const response = await client.messages.create({
      model: "claude-opus-5",
      max_tokens: 8192,
      system: SYSTEM_PROMPT,
      // Tools are withheld on the final turn so the loop cannot end with the
      // model asking for one more search it will never get. It has to answer
      // from what it already measured.
      ...(lastTurn ? {} : { tools: TOOLS as unknown as AnthropicSDK.Tool[] }),
      output_config: {
        effort: "medium",
        format: { type: "json_schema", schema: REPORT_SCHEMA },
      },
      messages,
    });

    if (response.stop_reason === "refusal") {
      throw new Error("model declined to answer");
    }

    if (response.stop_reason === "tool_use") {
      messages.push({ role: "assistant", content: response.content });

      const results: AnthropicSDK.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type !== "tool_use") continue;
        const output = runTool(
          block.name,
          block.input as Record<string, unknown>,
          snapshots,
        );
        toolLog.push({ tool: block.name, input: block.input, result: output });
        results.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: JSON.stringify(output),
        });
      }

      // All results go back in one user message. Splitting them trains the
      // model out of requesting tools in parallel.
      messages.push({ role: "user", content: results });
      continue;
    }

    const text = response.content.find((b) => b.type === "text");
    if (!text || text.type !== "text") {
      throw new Error("model returned no report");
    }

    return {
      ...(JSON.parse(text.text) as Record<string, unknown>),
      toolCalls: toolLog.length,
      transcript: toolLog,
      turnsUsed: turn + 1,
    };
  }

  throw new Error(`investigation did not conclude within ${MAX_TURNS} turns`);
}

export { runTool, investigate, MAX_TURNS };
