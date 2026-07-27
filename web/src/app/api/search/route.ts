/**
 * Tactical clip retrieval.
 *
 * The division of labour is the same one the rest of this project is built
 * around, applied to search instead of narration. A language model is good at
 * turning "when did we play through the middle against a compact block" into a
 * *query*, and has no business deciding whether frame 137 satisfies it. So the
 * model's entire output is a structured `TacticalQuery`, and the search itself
 * runs in `lib/tactics/search.ts` over measurements the engine already computed.
 *
 * That is what makes a result trustworthy enough to show a coach. Every
 * returned passage provably has the property that was asked for, the same
 * question always returns the same passages, and each hit carries the numbers
 * that qualified it. A model asked to pick moments directly can do none of
 * those things, and would confidently return a passage that simply is not one.
 *
 * With no API key the endpoint still works: `parseQueryHeuristically` reads the
 * common phrasings, and the response says which path produced the query so the
 * caller can tell a parsed question from a guessed one.
 */

import { NextRequest } from "next/server";

import type { TacticalQuery } from "@/lib/tactics/search";

interface SearchRequest {
  question?: string;
  /** A caller may skip the language step entirely and pass filters directly. */
  query?: TacticalQuery;
}

/**
 * The query schema, as the model is allowed to express it.
 *
 * Every field is a threshold on something the engine measures. There is
 * deliberately no free-text field and no way to express "interesting", because
 * anything the model cannot ground in one of these numbers is something it
 * would be inventing.
 */
const QUERY_SCHEMA = {
  type: "object",
  properties: {
    minOpenLanes: { type: "integer", minimum: 0, maximum: 10 },
    minBestMarginS: { type: "number", minimum: 0, maximum: 5 },
    minProgressionM: { type: "number", minimum: 0, maximum: 100 },
    minBlockWidthM: { type: "number", minimum: 0, maximum: 68 },
    maxBlockWidthM: { type: "number", minimum: 0, maximum: 68 },
    minBlockDepthM: { type: "number", minimum: 0, maximum: 105 },
    maxBlockDepthM: { type: "number", minimum: 0, maximum: 105 },
    minBackLineGapM: { type: "number", minimum: 0, maximum: 68 },
    minPlayersBetweenLines: { type: "integer", minimum: 0, maximum: 11 },
    minPlayersBeyondLine: { type: "integer", minimum: 0, maximum: 11 },
    attackingTeam: { type: "string", enum: ["team_a", "team_b"] },
  },
  additionalProperties: false,
  required: [],
} as const;

const SYSTEM_PROMPT = `You translate a football analyst's question into a structured search over tracking data.

You are NOT answering the question and NOT choosing moments. You only choose filters. Code runs the search.

The pitch is 105m by 68m. Useful reference points:
- A compact defensive block is under about 30m wide; a stretched one is over 45m.
- A low block sits under about 20m deep; a high line stretches beyond 30m.
- A gap over about 12m in the last line is a large one.
- A passing lane is "open" when the ball beats the nearest defender to it; a margin over 0.5s is comfortable.

Set only the filters the question actually implies. Every filter you add narrows the results, so inventing thresholds the analyst did not ask for is how a search returns nothing. If the question is vague, prefer one or two loose filters over many tight ones.

Write no prose. Return only the query object.`;

/** Phrasings common enough to be worth handling without a model. */
function parseQueryHeuristically(question: string): TacticalQuery {
  const q = question.toLowerCase();
  const query: TacticalQuery = {};

  // Pull an explicit distance when the analyst gives one, e.g. "wider than 40m".
  const metres = q.match(/(\d+(?:\.\d+)?)\s*m\b/);
  const value = metres ? Number(metres[1]) : null;

  if (/\b(compact|narrow|tight)\b/.test(q)) query.maxBlockWidthM = value ?? 32;
  if (/\b(stretch|wide|spread|pulled apart)\w*\b/.test(q))
    query.minBlockWidthM = value ?? 42;
  if (/\b(low block|deep|sat off|dropped)\b/.test(q)) query.maxBlockDepthM = value ?? 22;
  if (/\b(high line|pushed up|aggressive)\b/.test(q)) query.minBlockDepthM = value ?? 30;
  if (/\bgap\b/.test(q)) query.minBackLineGapM = value ?? 12;
  if (/\bbetween the lines\b/.test(q)) query.minPlayersBetweenLines = 1;
  if (/\b(beyond|behind|in behind|through ball|offside)\b/.test(q))
    query.minPlayersBeyondLine = 1;
  if (/\b(open|available|free)\b/.test(q)) query.minOpenLanes = 1;
  if (/\bprogress\w*|forward|up the pitch\b/.test(q)) query.minProgressionM = 10;

  // Never return a query with no filters at all: that matches every frame and
  // reads as a broken search rather than as an unparsed question.
  if (!Object.keys(query).length) query.minOpenLanes = 1;
  return query;
}

export async function POST(request: NextRequest) {
  let body: SearchRequest;
  try {
    body = (await request.json()) as SearchRequest;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }

  // An explicit query bypasses the language step. Useful for saved searches and
  // for a UI that offers filters directly.
  if (body.query) {
    return Response.json({ query: body.query, source: "explicit" });
  }

  const question = body.question?.trim();
  if (!question) {
    return Response.json(
      { error: "provide `question` or `query`" },
      { status: 400 },
    );
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return Response.json({
      query: parseQueryHeuristically(question),
      source: "heuristic",
    });
  }

  try {
    const query = await llmQuery(question, apiKey);
    return Response.json({ query, source: "claude" });
  } catch (error) {
    console.error("query translation via Claude failed, using heuristics:", error);
    return Response.json({
      query: parseQueryHeuristically(question),
      source: "heuristic-fallback",
    });
  }
}

async function llmQuery(question: string, apiKey: string): Promise<TacticalQuery> {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ apiKey });

  const response = await client.messages.create({
    model: "claude-opus-5",
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    // Low effort for the same reason as the narration route: this is a
    // translation task with a tight schema, not a reasoning one, and a search
    // box that stalls for several seconds is a worse tool than a fast one.
    output_config: {
      effort: "low",
      format: { type: "json_schema", schema: QUERY_SCHEMA },
    },
    messages: [{ role: "user", content: question }],
  });

  if (response.stop_reason === "refusal") {
    throw new Error("model declined to translate the question");
  }

  const text = response.content.find((block) => block.type === "text");
  if (!text || text.type !== "text") {
    throw new Error("model returned no text block");
  }

  const parsed = JSON.parse(text.text) as TacticalQuery;

  // The schema constrains types but not coherence. A query whose bounds cross
  // can never match anything, and would look to the analyst like "there were no
  // such moments" rather than like a bad translation.
  if (
    parsed.minBlockWidthM !== undefined &&
    parsed.maxBlockWidthM !== undefined &&
    parsed.minBlockWidthM > parsed.maxBlockWidthM
  ) {
    throw new Error("model produced contradictory width bounds");
  }
  if (
    parsed.minBlockDepthM !== undefined &&
    parsed.maxBlockDepthM !== undefined &&
    parsed.minBlockDepthM > parsed.maxBlockDepthM
  ) {
    throw new Error("model produced contradictory depth bounds");
  }

  return parsed;
}

export { parseQueryHeuristically };
