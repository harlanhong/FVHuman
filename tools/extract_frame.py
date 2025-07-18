import ffmpeg
import os
import glob
def extract_frames(video_path, output_path, frame_rate=1):
    ffmpeg.input(video_path).output(output_path, vf=f"fps={frame_rate}").run()

if __name__ == "__main__":
    fp = './data/ted_test_set2'
    for video_path in glob.glob(os.path.join(fp, '*.mp4')):
        video_name = os.path.basename(video_path)
        extract_frames(video_path, os.path.join(fp, f'{video_name}.jpg'))