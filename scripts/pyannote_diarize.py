#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-speakers", type=int, default=0)
    args = parser.parse_args()

    from pyannote.audio import Pipeline  # type: ignore

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()) and hasattr(pipeline, "to"):
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    kwargs = {}
    if int(args.num_speakers or 0) > 0:
        kwargs["num_speakers"] = int(args.num_speakers)

    diar = pipeline(args.audio, **kwargs)
    rows = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        rows.append(
            {
                "start_ms": int(float(turn.start) * 1000),
                "end_ms": int(float(turn.end) * 1000),
                "speaker_id": str(speaker),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
