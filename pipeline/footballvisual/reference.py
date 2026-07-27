"""Compare this pipeline against professional broadcast tracking.

Every accuracy number this repository reports is measured on its own synthetic
clip, which makes them honest but not *comparable*. "74.6% detection coverage"
sounds poor without knowing what is achievable: a reader has no way to tell
whether the remaining quarter is a defect in this pipeline or a property of
broadcast footage, where players are occluded, off frame, or fifteen pixels tall.

SkillCorner and PySport publish ten matches of broadcast tracking data produced
by a commercial system that clubs actually buy. Critically, each player carries
an `is_detected` flag separating the frames where the player was genuinely seen
from the frames where their position was extrapolated. That flag is the
benchmark: it says what fraction of players a state of the art system recovers
from a broadcast frame, and it turns this project's numbers from an absolute
claim into a relative one.

The data is tracking output, not video, so this cannot run the pipeline over it
and score positions player by player. What it can do is compare the summary
statistics that both systems produce, which is enough to answer the question a
reader actually has: are these numbers in the right range for the problem.

Download it with `make skillcorner`, or by hand from
https://github.com/SkillCorner/opendata (the tracking files are Git LFS, so
fetch them from the media host rather than raw.githubusercontent.com).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReferenceStats:
    """What a professional broadcast tracker recovers from a real match."""

    source: str
    frames: int
    #: Players the provider reports per frame. Always the full squad, because
    #: unseen players are extrapolated rather than dropped.
    players_reported_median: float
    #: Players actually *seen* in the frame. This is the comparable number.
    players_detected_median: float
    players_detected_mean: float
    #: Fraction of the reported squad that was genuinely detected.
    detection_rate_mean: float
    detection_rate_median: float
    #: Fraction of frames where the ball was seen rather than inferred.
    ball_detection_rate: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "playersReportedMedian": round(self.players_reported_median, 1),
            "playersDetectedMedian": round(self.players_detected_median, 1),
            "playersDetectedMean": round(self.players_detected_mean, 2),
            "detectionRateMean": round(self.detection_rate_mean, 4),
            "detectionRateMedian": round(self.detection_rate_median, 4),
            "ballDetectionRate": round(self.ball_detection_rate, 4),
        }


def load_skillcorner_stats(path: Path, max_frames: int | None = None) -> ReferenceStats:
    """Summarise a SkillCorner `*_tracking_extrapolated.jsonl` file.

    Frames before kickoff carry a full `player_data` list of nulls, so a frame
    only counts once at least one player has a position. Counting them would
    dilute the detection rate with footage that is not of the match.
    """
    path = Path(path)
    reported: list[int] = []
    detected: list[int] = []
    rates: list[float] = []
    ball_seen = 0
    frames = 0

    with path.open() as handle:
        for line in handle:
            if max_frames is not None and frames >= max_frames:
                break
            row = json.loads(line)
            players = row.get("player_data") or []
            positioned = [p for p in players if p.get("x") is not None]
            if not positioned:
                continue

            frames += 1
            seen = sum(1 for p in positioned if p.get("is_detected"))
            reported.append(len(positioned))
            detected.append(seen)
            rates.append(seen / len(positioned))
            if (row.get("ball_data") or {}).get("is_detected"):
                ball_seen += 1

    if not frames:
        raise ValueError(f"{path} contained no frames with player positions")

    return ReferenceStats(
        source=path.name,
        frames=frames,
        players_reported_median=float(np.median(reported)),
        players_detected_median=float(np.median(detected)),
        players_detected_mean=float(np.mean(detected)),
        detection_rate_mean=float(np.mean(rates)),
        detection_rate_median=float(np.median(rates)),
        ball_detection_rate=ball_seen / frames,
    )


@dataclass
class PipelineStats:
    """The comparable numbers, read back out of this pipeline's own output."""

    source: str
    frames: int
    players_tracked_median: float
    players_tracked_mean: float
    ball_present_rate: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "playersTrackedMedian": round(self.players_tracked_median, 1),
            "playersTrackedMean": round(self.players_tracked_mean, 2),
            "ballPresentRate": round(self.ball_present_rate, 4),
        }


def load_pipeline_stats(tracks_path: Path) -> PipelineStats:
    """Summarise a `tracks.json` in the same terms as the reference.

    Note the asymmetry this cannot remove: `ball_present_rate` counts frames
    where a ball position was written, and this pipeline writes one while the
    ball tracker is coasting on prediction. The reference's flag means the ball
    was actually seen. So a ball rate near 100% here is not comparable to the
    reference's, and is a reason to distrust it rather than to celebrate it.
    """
    tracks_path = Path(tracks_path)
    data = json.loads(tracks_path.read_text())
    frames = data.get("frames") or []
    if not frames:
        raise ValueError(f"{tracks_path} contained no frames")

    counts = [len(f.get("players") or []) for f in frames]
    with_ball = sum(1 for f in frames if f.get("ball"))

    return PipelineStats(
        source=tracks_path.name,
        frames=len(frames),
        players_tracked_median=float(np.median(counts)),
        players_tracked_mean=float(np.mean(counts)),
        ball_present_rate=with_ball / len(frames),
    )


def format_comparison(reference: ReferenceStats, pipeline: PipelineStats | None) -> str:
    """A short report, written to be read rather than parsed."""
    lines = [
        "broadcast tracking, professional reference",
        f"  source                    {reference.source}",
        f"  frames                    {reference.frames}",
        f"  players reported/frame    {reference.players_reported_median:.0f}"
        "   (full squad, unseen players extrapolated)",
        f"  players DETECTED/frame    {reference.players_detected_median:.0f}"
        f"   (mean {reference.players_detected_mean:.1f})",
        f"  detection rate            {reference.detection_rate_mean * 100:.1f}%"
        f"   (median {reference.detection_rate_median * 100:.1f}%)",
        f"  ball actually seen        {reference.ball_detection_rate * 100:.1f}% of frames",
    ]

    if pipeline is not None:
        lines += [
            "",
            "this pipeline",
            f"  source                    {pipeline.source}",
            f"  frames                    {pipeline.frames}",
            f"  players tracked/frame     {pipeline.players_tracked_median:.0f}"
            f"   (mean {pipeline.players_tracked_mean:.1f})",
            f"  frames with a ball        {pipeline.ball_present_rate * 100:.1f}%"
            "   (includes coasting, so not comparable)",
        ]

    lines += [
        "",
        "Read this as context, not as a score. The reference is a different match,",
        "on footage this repository does not have, so nothing here is a head to head.",
        "What it establishes is the range: recovering roughly half to two thirds of",
        "the squad from a broadcast frame is what a commercial system achieves, so a",
        "pipeline in that range is not obviously broken, and one reporting a hundred",
        "percent of anything is measuring something other than what it saw.",
    ]
    return "\n".join(lines)
