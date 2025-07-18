from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

# 输入视频路径
video_path = "/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/user_test/demo-video.mp4"
output_folder = "./output_clips"  # 输出切割视频的文件夹

# 创建 VideoManager 和 SceneManager
video_manager = VideoManager([video_path])
scene_manager = SceneManager()

# 添加内容检测器（你可以调整 threshold 参数）
scene_manager.add_detector(ContentDetector(threshold=30.0))

# 启动 VideoManager
video_manager.set_downscale_factor()
video_manager.start()

# 检测场景变化
scene_manager.detect_scenes(frame_source=video_manager)

# 获取检测到的场景时间戳
scene_list = scene_manager.get_scene_list()
print(f"Detected {len(scene_list)} scenes!")

# 打印每个场景的开始和结束时间
for i, scene in enumerate(scene_list):
    print(f"Scene {i + 1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")

# 使用 FFMpeg 将每个场景切割为独立视频
split_video_ffmpeg(video_path, scene_list, output_folder, arg_override='-c:v libx264 -crf 23 -preset veryfast')

# 释放资源
video_manager.release()
