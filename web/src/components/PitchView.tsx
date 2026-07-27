"use client";

/**
 * The top-down tactical map: the sandbox itself.
 *
 * Rendering is split across two layers for a reason. The space-control field is
 * a per-cell scalar, which is a canvas job, and drawing it as thousands of SVG
 * rects would stall the browser on every drag. Everything else is SVG so that
 * players are real DOM nodes with real pointer targets, which makes dragging
 * and hit testing the browser's problem rather than this component's.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { makeTransform, pitchLines, pitchSpots } from "@/lib/pitch";
import type { Arrow, DragOverride } from "@/lib/session";
import type { PassingLane, PlayerState, TacticalReport, Vec2 } from "@/lib/tactics";

export type PitchTool = "select" | "arrow";

interface Props {
  report: TacticalReport | null;
  players: PlayerState[];
  ball: Vec2 | null;
  paused: boolean;
  tool: PitchTool;
  arrows: Arrow[];
  selectedLaneId: number | null;
  showSpace: boolean;
  showLanes: boolean;
  showShape: boolean;
  onDragPlayer: (override: DragOverride) => void;
  onAddArrow: (arrow: Arrow) => void;
  onSelectPlayer: (id: number | null) => void;
}

const TEAM_FILL: Record<string, string> = {
  team_a: "#3b82f6",
  team_b: "#ef4444",
  // Keepers and officials are drawn distinctly because they are excluded from
  // the team shape metrics, and a reader should be able to see that the block
  // being measured does not include them.
  keeper: "#22d3ee",
  referee: "#facc15",
  other: "#a78bfa",
  unknown: "#94a3b8",
};

const LANE_STROKE: Record<string, string> = {
  open: "#22c55e",
  contested: "#f59e0b",
  blocked: "#6b7280",
};

export default function PitchView({
  report,
  players,
  ball,
  paused,
  tool,
  arrows,
  selectedLaneId,
  showSpace,
  showLanes,
  showShape,
  onDragPlayer,
  onAddArrow,
  onSelectPlayer,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: 900, height: 620 });
  const [dragging, setDragging] = useState<number | null>(null);
  const [arrowStart, setArrowStart] = useState<Vec2 | null>(null);
  const [arrowEnd, setArrowEnd] = useState<Vec2 | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      // The pitch is 105 by 68, so hold that ratio rather than filling the box
      // and stretching every distance on the map.
      const width = Math.max(320, rect.width);
      setSize({ width, height: Math.round((width * 68) / 105) });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const transform = useMemo(
    () => makeTransform(size.width, size.height, 18),
    [size.width, size.height],
  );

  const toPitchFromEvent = useCallback(
    (clientX: number, clientY: number): Vec2 => {
      const svg = svgRef.current;
      if (!svg) return { x: 0, y: 0 };
      const rect = svg.getBoundingClientRect();
      return transform.toPitch(clientX - rect.left, clientY - rect.top);
    },
    [transform],
  );

  // Space control heatmap.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = size.width;
    canvas.height = size.height;
    ctx.clearRect(0, 0, size.width, size.height);

    const space = report?.space;
    if (!showSpace || !space) return;

    // Draw into a small offscreen buffer at grid resolution and let the browser
    // scale it up smoothly. Filling cell rects directly produces visible
    // blockiness and is markedly slower.
    const buffer = document.createElement("canvas");
    buffer.width = space.cols;
    buffer.height = space.rows;
    const bctx = buffer.getContext("2d");
    if (!bctx) return;

    const image = bctx.createImageData(space.cols, space.rows);
    for (let r = 0; r < space.rows; r++) {
      for (let c = 0; c < space.cols; c++) {
        // Grid row 0 is the -y touchline; screen row 0 is +y. Flip on write.
        const v = space.grid[(space.rows - 1 - r) * space.cols + c];
        const i = (r * space.cols + c) * 4;
        const strength = Math.min(1, Math.abs(v));
        if (v < 0) {
          image.data[i] = 59;
          image.data[i + 1] = 130;
          image.data[i + 2] = 246;
        } else {
          image.data[i] = 239;
          image.data[i + 1] = 68;
          image.data[i + 2] = 68;
        }
        image.data[i + 3] = Math.round(strength * 86);
      }
    }
    bctx.putImageData(image, 0, 0);

    const { sx: x0, sy: y0 } = transform.toScreen(-52.5, 34);
    const { sx: x1, sy: y1 } = transform.toScreen(52.5, -34);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(buffer, x0, y0, x1 - x0, y1 - y0);
  }, [report, showSpace, size.width, size.height, transform]);

  const handlePointerDown = (event: React.PointerEvent) => {
    if (tool !== "arrow") return;
    const p = toPitchFromEvent(event.clientX, event.clientY);
    setArrowStart(p);
    setArrowEnd(p);
    (event.target as Element).setPointerCapture?.(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent) => {
    if (tool === "arrow" && arrowStart) {
      setArrowEnd(toPitchFromEvent(event.clientX, event.clientY));
      return;
    }
    if (dragging != null) {
      const p = toPitchFromEvent(event.clientX, event.clientY);
      onDragPlayer({ id: dragging, x: p.x, y: p.y });
    }
  };

  const handlePointerUp = () => {
    if (tool === "arrow" && arrowStart && arrowEnd) {
      const len = Math.hypot(arrowEnd.x - arrowStart.x, arrowEnd.y - arrowStart.y);
      // Ignore taps. Without this every stray click leaves a zero-length arrow
      // that is impossible to see and impossible to select in order to delete.
      if (len > 1.5) {
        onAddArrow({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          from: arrowStart,
          to: arrowEnd,
        });
      }
    }
    setArrowStart(null);
    setArrowEnd(null);
    setDragging(null);
  };

  const shownLanes = useMemo(() => {
    const lanes = showLanes ? report?.lanes ?? [] : [];
    if (selectedLaneId != null) return lanes.filter((l) => l.targetId === selectedLaneId);
    // Showing all twenty-odd lanes at once is unreadable. The best few carry
    // the decision, and any single lane can still be isolated by selecting it.
    return lanes.slice(0, 5);
  }, [report, showLanes, selectedLaneId]);

  const line = (pts: Array<[number, number]>) =>
    pts
      .map(([x, y]) => {
        const { sx, sy } = transform.toScreen(x, y);
        return `${sx.toFixed(1)},${sy.toFixed(1)}`;
      })
      .join(" ");

  return (
    <div ref={wrapRef} className="relative w-full select-none">
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 rounded-lg"
        style={{ width: size.width, height: size.height }}
      />
      <svg
        ref={svgRef}
        width={size.width}
        height={size.height}
        className={`relative rounded-lg bg-emerald-950/60 ring-1 ring-white/10 ${
          tool === "arrow" ? "cursor-crosshair" : "cursor-default"
        }`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 7 3.5, 0 7" fill="#fbbf24" />
          </marker>
          <marker id="laneend" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <polygon points="0 0, 6 3, 0 6" fill="currentColor" />
          </marker>
        </defs>

        {pitchLines().map((pts, i) => (
          <polyline
            key={i}
            points={line(pts)}
            fill="none"
            stroke="rgba(255,255,255,0.34)"
            strokeWidth={1.4}
          />
        ))}
        {pitchSpots().map(([x, y], i) => {
          const { sx, sy } = transform.toScreen(x, y);
          return <circle key={i} cx={sx} cy={sy} r={2} fill="rgba(255,255,255,0.4)" />;
        })}

        {showShape && report?.block && report.block.hull.length > 2 && (
          <polygon
            points={line(report.block.hull.map((p) => [p.x, p.y]) as Array<[number, number]>)}
            fill="rgba(239,68,68,0.10)"
            stroke="rgba(239,68,68,0.55)"
            strokeWidth={1.2}
            strokeDasharray="5 4"
          />
        )}

        {showShape && report?.block && (
          <>
            {[
              { x: report.block.defensiveLineX, label: "def line", colour: "#ef4444" },
              { x: report.block.offsideLineX, label: "offside", colour: "#fbbf24" },
            ].map((l, i) => {
              const a = transform.toScreen(l.x, -34);
              const b = transform.toScreen(l.x, 34);
              return (
                <g key={i}>
                  <line
                    x1={a.sx}
                    y1={a.sy}
                    x2={b.sx}
                    y2={b.sy}
                    stroke={l.colour}
                    strokeWidth={1}
                    strokeDasharray="3 5"
                    opacity={0.75}
                  />
                  {/*
                    Stacked, not both on the baseline. The offside line sits at
                    the second-deepest defender and the defensive line at the
                    deepest, so the two are a metre or two apart in the normal
                    case and their labels overlap into an unreadable smear. That
                    is the common case here rather than an edge one.
                  */}
                  <text
                    x={a.sx + 3}
                    y={a.sy - 5 - i * 11}
                    fill={l.colour}
                    fontSize={9}
                    opacity={0.85}
                  >
                    {l.label}
                  </text>
                </g>
              );
            })}
          </>
        )}

        {shownLanes.map((lane) => (
          <LaneGraphic key={lane.targetId} lane={lane} transform={transform} />
        ))}

        {arrows.map((a) => {
          const from = transform.toScreen(a.from.x, a.from.y);
          const to = transform.toScreen(a.to.x, a.to.y);
          return (
            <line
              key={a.id}
              x1={from.sx}
              y1={from.sy}
              x2={to.sx}
              y2={to.sy}
              stroke="#fbbf24"
              strokeWidth={2.2}
              markerEnd="url(#arrowhead)"
            />
          );
        })}

        {arrowStart && arrowEnd && (
          <line
            x1={transform.toScreen(arrowStart.x, arrowStart.y).sx}
            y1={transform.toScreen(arrowStart.x, arrowStart.y).sy}
            x2={transform.toScreen(arrowEnd.x, arrowEnd.y).sx}
            y2={transform.toScreen(arrowEnd.x, arrowEnd.y).sy}
            stroke="#fbbf24"
            strokeWidth={2}
            strokeDasharray="4 3"
            markerEnd="url(#arrowhead)"
          />
        )}

        {players.map((p) => {
          const { sx, sy } = transform.toScreen(p.x, p.y);
          const isCarrier = report?.carrierId === p.id;
          const beyond = report?.playersBeyondLine.includes(p.id);
          const between = report?.playersBetweenLines.includes(p.id);
          return (
            <g
              key={p.id}
              transform={`translate(${sx.toFixed(1)},${sy.toFixed(1)})`}
              className={paused ? "cursor-grab" : "cursor-pointer"}
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelectPlayer(p.id);
                // Dragging is only allowed while paused. Letting a drag land
                // during playback would be overwritten by the next frame a few
                // milliseconds later, which reads as the app ignoring you.
                if (paused && tool === "select") {
                  setDragging(p.id);
                  (e.target as Element).setPointerCapture?.(e.pointerId);
                }
              }}
            >
              {between && <circle r={11} fill="none" stroke="#22c55e" strokeWidth={1.4} opacity={0.9} />}
              {beyond && <circle r={13} fill="none" stroke="#fbbf24" strokeWidth={1} opacity={0.8} />}
              <circle
                r={isCarrier ? 8 : 6.5}
                fill={TEAM_FILL[p.team] ?? TEAM_FILL.unknown}
                stroke={isCarrier ? "#ffffff" : "rgba(0,0,0,0.55)"}
                strokeWidth={isCarrier ? 2 : 1}
              />
              <text
                y={3.2}
                textAnchor="middle"
                fontSize={7.5}
                fontWeight={600}
                fill="white"
                pointerEvents="none"
              >
                {p.label ?? p.id}
              </text>
            </g>
          );
        })}

        {ball && (
          <circle
            cx={transform.toScreen(ball.x, ball.y).sx}
            cy={transform.toScreen(ball.x, ball.y).sy}
            r={4}
            fill="#ffffff"
            stroke="#111827"
            strokeWidth={1.2}
          />
        )}
      </svg>
    </div>
  );
}

function LaneGraphic({
  lane,
  transform,
}: {
  lane: PassingLane;
  transform: ReturnType<typeof makeTransform>;
}) {
  const from = transform.toScreen(lane.from.x, lane.from.y);
  const to = transform.toScreen(lane.to.x, lane.to.y);
  const colour = LANE_STROKE[lane.verdict];

  // Mark where the best-placed defender meets the lane. That point is the
  // actual reason for the verdict, so showing it makes the judgement auditable
  // instead of asking the user to trust a colour.
  const total = Math.hypot(lane.to.x - lane.from.x, lane.to.y - lane.from.y);
  const t = total > 0 ? lane.threatPointM / total : 0;
  const pinch = transform.toScreen(
    lane.from.x + (lane.to.x - lane.from.x) * t,
    lane.from.y + (lane.to.y - lane.from.y) * t,
  );

  return (
    <g style={{ color: colour }}>
      <line
        x1={from.sx}
        y1={from.sy}
        x2={to.sx}
        y2={to.sy}
        stroke={colour}
        strokeWidth={lane.verdict === "open" ? 2.4 : 1.6}
        strokeDasharray={lane.verdict === "blocked" ? "3 4" : undefined}
        opacity={lane.verdict === "blocked" ? 0.5 : 0.95}
        markerEnd="url(#laneend)"
      />
      {lane.verdict !== "open" && (
        <circle cx={pinch.sx} cy={pinch.sy} r={3} fill={colour} opacity={0.9} />
      )}
    </g>
  );
}
