from __future__ import annotations

from moviepy import VideoFileClip, concatenate_videoclips



def main() -> None:
    p1 = "artifacts/videos/1_untrained"
    p2 = "artifacts/videos/2_half"
    p3 = "artifacts/videos/3_final"

    # take the only mp4 in each folder
    import glob
    def pick(folder: str) -> str:
        mp4s = glob.glob(folder + "/*.mp4")
        mp4s.sort()
        return mp4s[-1]

    v1 = pick(p1)
    v2 = pick(p2)
    v3 = pick(p3)

    clips = [VideoFileClip(v1), VideoFileClip(v2), VideoFileClip(v3)]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile("artifacts/videos/evolution.mp4", fps=30)

    for c in clips:
        c.close()
    final.close()

    print("Saved: artifacts/videos/evolution.mp4")


if __name__ == "__main__":
    main()
