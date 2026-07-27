"use client";

/**
 * The sandbox.
 *
 * Playback and editing are deliberately mutually exclusive. While the clip is
 * running the map shows tracked reality and dragging is disabled, because any
 * edit would be overwritten by the next frame 40ms later. Pausing freezes the
 * frame and unlocks dragging, and from then on the map shows tracked reality
 * plus whatever the analyst has posed on top of it. The override layer is kept
 * separate from the loaded data (see `session.ts`) so "reset" is exact and the
 * analysis can always state whether it is describing the video or a hypothesis.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AgentPanel from "@/components/AgentPanel";
import ClipLoader from "@/components/ClipLoader";
import PitchView, { type PitchTool } from "@/components/PitchView";
import TacticsPanel from "@/components/TacticsPanel";
import {
  applyOverrides,
  formatTime,
  loadTracks,
  teamLabel,
  type Arrow,
  type DragOverride,
  type LoadedSession,
} from "@/lib/session";
import { analyseSnapshot, type TacticalReport } from "@/lib/tactics";

const TRACKS_URL = "/data/tracks.json";

/**
 * Playback speeds, defaulting to half.
 *
 * Real time is the wrong default for this tool. Twenty two dots moving at match
 * pace on a 105 by 68 metre map is genuinely hard to read: a passing lane opens
 * and closes inside a few frames, and the thing the analyst is here to watch is
 * gone before they have found it. Coaches step through video slowly for exactly
 * this reason. Full speed stays available, it just is not what the tool opens
 * on.
 */
const SPEEDS = [0.25, 0.5, 1] as const;
type Speed = (typeof SPEEDS)[number];

export default function SandboxPage() {
  const [session, setSession] = useState<LoadedSession | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tool, setTool] = useState<PitchTool>("select");
  const [overrides, setOverrides] = useState<Map<number, DragOverride>>(new Map());
  const [arrows, setArrows] = useState<Arrow[]>([]);
  const [selectedLane, setSelectedLane] = useState<number | null>(null);
  // Space control is the heaviest overlay, a shaded field over the whole pitch,
  // and it competes with the lanes drawn on top of it. Off by default; the
  // analyst turns it on when that is the question they are asking.
  const [showSpace, setShowSpace] = useState(false);
  const [showLanes, setShowLanes] = useState(true);
  const [showShape, setShowShape] = useState(true);
  const [speed, setSpeed] = useState<Speed>(0.5);
  // Focus mode shows the three best passing options and nothing else. Every
  // measurement is still one click away in Detail, but opening on all of them
  // at once buries the answer to the question the tool is actually for.
  const [detailed, setDetailed] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [clipName, setClipName] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    loadTracks(TRACKS_URL)
      .then((loaded) => {
        setSession(loaded);
        // Open on the first frame that actually has a squad on it. The tracker
        // needs a few frames of consistent detections before it will confirm a
        // track, so frame zero is legitimately empty and landing there shows a
        // bare pitch and an analysis panel with nothing to say.
        const first = loaded.snapshots.findIndex((s) => s.players.length >= 8);
        setIndex(first >= 0 ? first : 0);
      })
      .catch((e: Error) => setLoadError(e.message));
  }, []);

  const frameCount = session?.snapshots.length ?? 0;
  const fps = session?.meta.fps ?? 25;

  // Playback. A requestAnimationFrame loop with an explicit accumulator rather
  // than setInterval, so the map advances at the clip's real frame rate instead
  // of drifting with the browser's timer resolution.
  useEffect(() => {
    if (!playing || !frameCount) return;
    lastTickRef.current = performance.now();

    const step = (now: number) => {
      const elapsed = now - lastTickRef.current;
      const frameMs = 1000 / (fps * speed);
      if (elapsed >= frameMs) {
        const advance = Math.floor(elapsed / frameMs);
        lastTickRef.current += advance * frameMs;
        setIndex((i) => {
          const next = i + advance;
          if (next >= frameCount - 1) {
            setPlaying(false);
            return frameCount - 1;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, frameCount, fps, speed]);

  // The video has to be slowed by the same factor, or the two panes drift apart
  // and the seek correction below fights the playback rate every frame.
  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = speed;
  }, [speed, videoUrl]);

  // Keep the broadcast video aligned with the map. Only corrected when it has
  // drifted more than a couple of frames, because assigning currentTime every
  // frame makes the video stutter.
  //
  // The seek is deferred until the video has metadata. Setting `currentTime`
  // before then is silently discarded, which leaves the pane black on load
  // while the map beside it already shows the frame.
  useEffect(() => {
    const video = videoRef.current;
    const snapshot = session?.snapshots[index];
    if (!video || !snapshot) return;

    const seek = () => {
      if (Math.abs(video.currentTime - snapshot.timeS) > 2 / fps) {
        video.currentTime = snapshot.timeS;
      }
    };

    if (video.readyState >= 1) {
      seek();
    } else {
      video.addEventListener("loadedmetadata", seek, { once: true });
      return () => video.removeEventListener("loadedmetadata", seek);
    }
  }, [index, session, fps]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (playing) void video.play().catch(() => undefined);
    else video.pause();
  }, [playing]);

  const baseSnapshot = session?.snapshots[index] ?? null;
  const previousSnapshot = index > 0 ? session?.snapshots[index - 1] ?? null : null;

  const snapshot = useMemo(
    () => (baseSnapshot ? applyOverrides(baseSnapshot, overrides, null) : null),
    [baseSnapshot, overrides],
  );

  const report: TacticalReport | null = useMemo(() => {
    if (!snapshot) return null;
    // Possession stabilisation lives in the engine, not here: it derives the
    // fallback from `previous`, so this stays a pure function of the frame.
    return analyseSnapshot(snapshot, {
      previous: previousSnapshot,
      spaceCellM: 2,
      computeSpace: showSpace,
    });
  }, [snapshot, previousSnapshot, showSpace]);

  const handleDrag = useCallback((override: DragOverride) => {
    setOverrides((prev) => {
      const next = new Map(prev);
      next.set(override.id, override);
      return next;
    });
  }, []);

  const resetEdits = useCallback(() => {
    setOverrides(new Map());
    setArrows([]);
    setSelectedLane(null);
  }, []);

  const handleLoadClip = useCallback(
    (loaded: LoadedSession, url: string | null, name: string) => {
      // Everything derived from the old clip has to go, or a drag meant for
      // player 7 in one match silently moves player 7 in the next.
      setPlaying(false);
      setOverrides(new Map());
      setArrows([]);
      setSelectedLane(null);
      setSession(loaded);
      setVideoUrl(url);
      setClipName(name);
      const first = loaded.snapshots.findIndex((s) => s.players.length >= 8);
      setIndex(first >= 0 ? first : 0);
    },
    [],
  );

  const paused = !playing;
  const edited = overrides.size > 0;
  const evaluation = session?.meta.stats?.evaluation as
    | Record<string, number>
    | undefined;

  if (loadError) {
    return (
      <main className="mx-auto max-w-2xl p-8">
        <h1 className="mb-3 text-xl font-semibold text-slate-100">FootballVisual</h1>
        <div className="rounded-lg border border-red-500/30 bg-red-950/30 p-4 text-sm text-red-200">
          <p className="mb-2 font-semibold">Could not load tracking data.</p>
          <p className="mb-3 text-red-300/80">{loadError}</p>
          <p className="text-slate-300">Generate it first from the repository root:</p>
          <pre className="mt-2 overflow-x-auto rounded bg-slate-950 p-3 text-xs text-slate-300">
            make demo
          </pre>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[1600px] p-4 lg:p-6">
      <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">
            FootballVisual <span className="text-slate-500">tactical sandbox</span>
          </h1>
          <p className="text-xs text-slate-400">
            Broadcast video to a top-down tactical map, via detection, tracking and a
            planar homography. Pause to drag players and re-run the analysis.
          </p>
        </div>
        <div className="flex items-start gap-3">
          {session && (
            <div className="text-right text-[11px] text-slate-500">
              <div>
                {session.tracks.length} tracks · {frameCount} frames ·{" "}
                {fps.toFixed(0)} fps
              </div>
              {evaluation && (
                <div className="text-slate-600">
                  position MAE {evaluation.positionMaeM} m vs ground truth
                </div>
              )}
            </div>
          )}
          <ClipLoader onLoad={handleLoadClip} currentName={clipName} />
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-3">
          <div className="grid gap-3 xl:grid-cols-2">
            <div className="overflow-hidden rounded-lg border border-white/10 bg-black">
              {/*
                Two sources on purpose. H.264 is patent-encumbered, so
                open-source Chromium builds ship without it and report no
                support for avc1 while playing VP9 happily. Listing both lets
                each browser take the one it can decode.
              */}
              {videoUrl ? (
                // `key` forces a fresh element per clip. Swapping `src` on a
                // <video> that has already loaded leaves the old frames on
                // screen until the new one buffers.
                <video
                  key={videoUrl}
                  ref={videoRef}
                  src={videoUrl}
                  muted
                  playsInline
                  preload="auto"
                  className="w-full"
                />
              ) : (
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  preload="auto"
                  className="w-full"
                >
                  <source src="/data/broadcast.webm" type="video/webm" />
                  <source src="/data/broadcast.mp4" type="video/mp4" />
                </video>
              )}
              <div className="border-t border-white/10 px-3 py-1.5 text-[11px] text-slate-500">
                Source broadcast. Detection and tracking run on these pixels.
              </div>
            </div>

            <div>
              <PitchView
                report={report}
                players={snapshot?.players ?? []}
                ball={snapshot?.ball ?? null}
                paused={paused}
                tool={tool}
                arrows={arrows}
                selectedLaneId={selectedLane}
                showSpace={showSpace}
                showLanes={showLanes}
                showShape={showShape}
                onDragPlayer={handleDrag}
                onAddArrow={(a) => setArrows((prev) => [...prev, a])}
                onSelectPlayer={() => undefined}
              />
              <div className="mt-1 px-1 text-[11px] text-slate-500">
                Top-down map in pitch metres.{" "}
                {paused ? (
                  <span className="text-emerald-400">
                    Paused, drag players to pose a shape.
                  </span>
                ) : (
                  <span>Playing, pause to edit.</span>
                )}
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-white/10 bg-slate-900/50 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                className="rounded bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white transition hover:bg-emerald-500"
              >
                {playing ? "Pause" : "Play"}
              </button>
              <span className="w-20 font-mono text-xs text-slate-400">
                {formatTime(baseSnapshot?.timeS ?? 0)}
              </span>
              <input
                type="range"
                min={0}
                max={Math.max(0, frameCount - 1)}
                value={index}
                onChange={(e) => {
                  setPlaying(false);
                  setIndex(Number(e.target.value));
                }}
                className="min-w-[180px] flex-1 accent-emerald-500"
                aria-label="Timeline"
              />
              <span className="font-mono text-xs text-slate-500">
                {index + 1}/{frameCount}
              </span>
            </div>

            <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-500">Speed</span>
                {SPEEDS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSpeed(s)}
                    className={`rounded px-2 py-0.5 font-mono transition ${
                      speed === s
                        ? "bg-slate-200 text-slate-900"
                        : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setIndex((i) => Math.max(0, i - 1));
                  }}
                  className="rounded bg-slate-800 px-2 py-0.5 text-slate-300 transition hover:bg-slate-700"
                  aria-label="Previous frame"
                >
                  &larr; frame
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setIndex((i) => Math.min(frameCount - 1, i + 1));
                  }}
                  className="rounded bg-slate-800 px-2 py-0.5 text-slate-300 transition hover:bg-slate-700"
                  aria-label="Next frame"
                >
                  frame &rarr;
                </button>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-white/10 pt-3 text-xs">
              <div className="flex gap-1">
                {(["select", "arrow"] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTool(t)}
                    disabled={!paused}
                    className={`rounded px-2.5 py-1 transition disabled:opacity-40 ${
                      tool === t
                        ? "bg-slate-200 text-slate-900"
                        : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                    }`}
                  >
                    {t === "select" ? "Drag" : "Draw arrow"}
                  </button>
                ))}
              </div>

              <Toggle label="Lanes" checked={showLanes} onChange={setShowLanes} />
              <Toggle label="Block shape" checked={showShape} onChange={setShowShape} />
              <Toggle label="Space control" checked={showSpace} onChange={setShowSpace} />

              <button
                type="button"
                onClick={() => setDetailed((d) => !d)}
                className="rounded border border-white/15 bg-slate-800 px-2.5 py-1 text-slate-300 transition hover:bg-slate-700"
              >
                {detailed ? "Focus view" : "All measurements"}
              </button>

              {(edited || arrows.length > 0) && (
                <button
                  type="button"
                  onClick={resetEdits}
                  className="ml-auto rounded border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-amber-400 transition hover:bg-amber-500/20"
                >
                  Reset {edited ? `${overrides.size} moved` : "annotations"}
                </button>
              )}
            </div>

            {edited && (
              <p className="mt-2 text-[11px] text-amber-400/80">
                Showing a posed shape. {overrides.size} player
                {overrides.size > 1 ? "s have" : " has"} been moved from where the video
                had them, and the analysis describes the hypothesis.
              </p>
            )}
          </div>

          {report && (
            <div className="flex flex-wrap gap-3 rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2 text-[11px] text-slate-400">
              <span>
                In possession:{" "}
                <span className="text-slate-200">{teamLabel(report.attackingTeam)}</span>
              </span>
              <span>
                Carrier:{" "}
                <span className="text-slate-200">
                  {report.carrierId != null ? `#${report.carrierId}` : "none"}
                </span>
              </span>
              <span>
                Open lanes:{" "}
                <span className="text-emerald-400">
                  {report.lanes.filter((l) => l.verdict === "open").length}
                </span>{" "}
                of {report.lanes.length}
              </span>
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <AgentPanel report={report} edited={edited} />
          <TacticsPanel
            report={report}
            selectedLaneId={selectedLane}
            onSelectLane={setSelectedLane}
            detailed={detailed}
          />
        </aside>
      </div>
    </main>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 text-slate-300">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-emerald-500"
      />
      {label}
    </label>
  );
}
