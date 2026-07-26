/**
 * Tactical narration endpoint.
 *
 * The division of labour here is the point of the whole project. Every
 * tactical *fact* arrives already computed: which lanes are open, by how many
 * seconds, how compact the block is, where the gap in the back line sits. The
 * language model's only job is to turn those numbers into the sentence a coach
 * would say. It never sees raw coordinates, so it cannot redo the geometry and
 * reach a different answer to the one drawn on screen, and it is never asked
 * which lane is open.
 *
 * That constraint is what makes the analysis reproducible. It also means the
 * deterministic fallback below is not a degraded stub: it has access to exactly
 * the same facts the model does, so with no API key configured the product
 * still says something true and specific. It just says it less fluently.
 */

import { NextRequest } from "next/server";

import type { LaneVerdict } from "@/lib/tactics";

/** The fact shape produced by `reportFacts()` in the tactics engine. */
interface LaneFact {
  target: string | null;
  verdict: LaneVerdict;
  safetyMarginS: number;
  distanceM: number;
  progressionM: number;
  defendersBypassed: number;
  receiverPressureM: number;
  closestDefender: string | null;
}

interface Facts {
  attackingTeam: string;
  defendingTeam: string;
  ballCarrier: string | null;
  lanes: LaneFact[];
  defensiveBlock: {
    players: number;
    hullAreaM2: number;
    widthM: number;
    blockDepthM: number;
    defensiveLineX: number;
    largestBackLineGapM: number;
    gapCentreY: number;
  } | null;
  spaceControl: { teamAShare: number; teamBShare: number } | null;
  playersBetweenLines: (string | null)[];
  playersBeyondLine: (string | null)[];
}

interface AnalyseRequest {
  facts: Facts;
  question?: string;
  /** True when the user has dragged players, so positions are hypothetical. */
  edited?: boolean;
}

const SYSTEM_PROMPT = `You are a football (soccer) tactical analyst reading a live top-down tracking map.

You are given measurements already computed from the tracking data by a deterministic geometry engine. Treat them as ground truth and build your answer on them.

The single most important rule: you do not decide which passing lane is open. That has already been computed. "safetyMarginS" is the number of seconds by which the ball beats the best placed defender to the most dangerous point on the pass. Positive means the pass gets there first. Negative means it gets cut out. Report what the numbers say, never your own guess.

How to write:
- Answer in 2 to 4 sentences of plain, spoken analysis, the way a coach talks to a player.
- Cite the specific number that justifies each claim, in seconds or metres.
- Refer to players by the label given.
- Never use em dashes, en dashes, or " - " as sentence punctuation. Use a comma, "and", or a second sentence.
- Do not invent players, positions, or events that are not in the measurements.
- If the best available option is still a poor one, say so plainly rather than talking it up.
- Do not include XML tags or internal reasoning in your answer.`;

const RESPONSE_SCHEMA = {
  type: "object" as const,
  properties: {
    headline: {
      type: "string" as const,
      description: "One short clause naming the single best option, under 12 words.",
    },
    analysis: {
      type: "string" as const,
      description: "2 to 4 sentences of tactical analysis citing the measurements.",
    },
    recommendedTarget: {
      type: ["string", "null"] as const,
      description: "Label of the player who should receive the ball, or null.",
    },
  },
  required: ["headline", "analysis", "recommendedTarget"],
  additionalProperties: false,
};

export async function POST(request: NextRequest) {
  let body: AnalyseRequest;
  try {
    body = (await request.json()) as AnalyseRequest;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }

  if (!body?.facts) {
    return Response.json({ error: "missing `facts` in request body" }, { status: 400 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return Response.json({ ...deterministicAnalysis(body), source: "deterministic" });
  }

  try {
    const result = await llmAnalysis(body, apiKey);
    return Response.json({ ...result, source: "claude" });
  } catch (error) {
    // Any model failure degrades to the deterministic answer rather than
    // surfacing an error. A tactical sandbox that goes blank because a network
    // call timed out is worse than one that speaks plainly.
    console.error("tactical analysis via Claude failed, using deterministic path:", error);
    return Response.json({ ...deterministicAnalysis(body), source: "deterministic-fallback" });
  }
}

async function llmAnalysis(body: AnalyseRequest, apiKey: string) {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ apiKey });

  const question =
    body.question?.trim() ||
    "Analyse this defensive block. Where is the open passing lane?";

  const userContent = [
    body.edited
      ? "Note: the analyst has dragged one or more players, so this is a hypothetical shape, not the tracked frame. Analyse it as posed."
      : "This is the tracked frame, paused.",
    "",
    `Question: ${question}`,
    "",
    "Computed measurements:",
    JSON.stringify(body.facts, null, 2),
  ].join("\n");

  const response = await client.messages.create({
    model: "claude-opus-5",
    max_tokens: 2048,
    system: SYSTEM_PROMPT,
    // Effort is kept low deliberately. The reasoning has already been done by
    // the geometry engine; what remains is phrasing, and a sandbox that pauses
    // for many seconds before speaking is a worse tool than one that answers
    // immediately.
    output_config: {
      effort: "low",
      format: { type: "json_schema", schema: RESPONSE_SCHEMA },
    },
    messages: [{ role: "user", content: userContent }],
  });

  if (response.stop_reason === "refusal") {
    throw new Error("model declined to answer");
  }

  const text = response.content.find((block) => block.type === "text");
  if (!text || text.type !== "text") {
    throw new Error("model returned no text block");
  }

  const parsed = JSON.parse(text.text) as {
    headline: string;
    analysis: string;
    recommendedTarget: string | null;
  };
  return parsed;
}

/**
 * Rule-based analysis from the same facts the model gets.
 *
 * This runs whenever there is no API key or the model call fails, and it is
 * written to be genuinely useful rather than a placeholder. Everything it says
 * is drawn from a measurement, which is the same standard the model is held to.
 */
export function deterministicAnalysis(body: AnalyseRequest): {
  headline: string;
  analysis: string;
  recommendedTarget: string | null;
} {
  const { facts } = body;
  const lanes = facts.lanes ?? [];
  const open = lanes.filter((l) => l.verdict === "open");
  const contested = lanes.filter((l) => l.verdict === "contested");

  if (!facts.ballCarrier) {
    return {
      headline: "No clear ball carrier in this frame",
      analysis:
        "Nobody on the attacking side is close enough to the ball to be carrying it, so there are no passing options to rank yet. Scrub to a frame where a player is on the ball, or drag one onto it to pose a situation.",
      recommendedTarget: null,
    };
  }

  const best = lanes[0] ?? null;
  const sentences: string[] = [];

  if (open.length) {
    const b = open[0];
    sentences.push(
      `${facts.ballCarrier} has ${open.length} clean option${open.length > 1 ? "s" : ""}, and the best is ${b.target}, ${b.distanceM.toFixed(0)} metres away with ${b.safetyMarginS.toFixed(2)} seconds of margin over ${b.closestDefender ?? "the nearest defender"}.`,
    );
    if (b.defendersBypassed > 0) {
      sentences.push(
        `That pass takes ${b.defendersBypassed} defender${b.defendersBypassed > 1 ? "s" : ""} out of the game and moves the ball ${b.progressionM.toFixed(0)} metres up the pitch.`,
      );
    }
    if (b.receiverPressureM < 5) {
      sentences.push(
        `The lane is open but the reception is not, ${b.target} has a defender ${b.receiverPressureM.toFixed(1)} metres away, so it needs to be played into the safe side.`,
      );
    }
  } else if (contested.length) {
    const c = contested[0];
    sentences.push(
      `Nothing is properly open. The closest to it is ${c.target}, where the ball beats ${c.closestDefender ?? "the nearest defender"} by only ${c.safetyMarginS.toFixed(2)} seconds, so it is a pass that has to be struck early and hard.`,
    );
  } else if (best) {
    sentences.push(
      `Every lane is covered. The least bad is ${best.target}, and even there ${best.closestDefender ?? "a defender"} gets across ${Math.abs(best.safetyMarginS).toFixed(2)} seconds before the ball arrives, so it would be cut out.`,
    );
  }

  const block = facts.defensiveBlock;
  if (block) {
    const compact = block.blockDepthM < 25;
    sentences.push(
      `The ${facts.defendingTeam} block is ${compact ? "compact" : "stretched"}, ${block.blockDepthM.toFixed(0)} metres front to back and ${block.widthM.toFixed(0)} across, covering ${block.hullAreaM2} square metres.`,
    );
    if (block.largestBackLineGapM > 12) {
      sentences.push(
        `Its weak point is a ${block.largestBackLineGapM.toFixed(0)} metre gap in the back line around y = ${block.gapCentreY.toFixed(0)}, which is where a runner should be attacking.`,
      );
    }
  }

  const between = facts.playersBetweenLines?.filter(Boolean) ?? [];
  if (between.length) {
    sentences.push(
      `${between.join(" and ")} ${between.length > 1 ? "are" : "is"} already between the lines, which is the space this block exists to deny.`,
    );
  }

  const headline = open.length
    ? `Play it to ${open[0].target}, ${open[0].safetyMarginS.toFixed(2)}s of margin`
    : contested.length
      ? `Nothing clean, ${contested[0].target} is the tightest option`
      : "Every lane is covered, hold or reset";

  return {
    headline,
    analysis: sentences.join(" "),
    recommendedTarget: open[0]?.target ?? contested[0]?.target ?? null,
  };
}
