import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter1d
import argparse
import pdb
def draw_annotations_for_extended_frames(video_batch, start_index_prediction=17):
    """
    video_batch     List of list of PIL.Image frames
    """
    radius = 2.5
    offset = 10
    for video in video_batch:
        assert start_index_prediction < len(video), f"Index {start_index_prediction} is out-of-bound for frames"
        for i_idx, image in enumerate(video):
            if i_idx < start_index_prediction:
                continue
            draw = ImageDraw.Draw(image)
            draw.ellipse([offset, offset, offset+2*radius, offset+2*radius], fill=(255,0,0))
    return video_batch

def draw_annotations_for_initial_frames(video_batch, end_index_prediction=17):
    """
    video_batch     List of list of PIL.Image frames
    """
    radius = 2.5
    offset = 10
    for video in video_batch:
        assert end_index_prediction < len(video), f"Index {end_index_prediction} is out-of-bound for frames"
        for i_idx, image in enumerate(video):
            if i_idx >= end_index_prediction:
                continue
            draw = ImageDraw.Draw(image)
            draw.ellipse([offset, offset, offset+2*radius, offset+2*radius], fill=(255,0,0))
    return video_batch

def images_to_array(images):
    return np.array([np.array(img) for img in images])

def array_to_images(array):
    return [Image.fromarray(arr) for arr in array]

def save_video_mp4(path, video, fps=12):
    imageio.mimwrite(
        path,
        video,
        format="mp4",
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

def blend_pixels_temporal(video_batch, start_index_prediction=17, sigma=1, support=3):
    for video in video_batch:
        assert start_index_prediction < len(video) and start_index_prediction > 0, f"Index {start_index_prediction} is out-of-bound for frames"
        # blur temporally
        video_array = images_to_array(video)
        start = max(start_index_prediction - support // 2, 0)
        end = min(start_index_prediction + support // 2 + 1, video_array.shape[0])
        # only blend in the first frame
        video_array[start_index_prediction] = gaussian_filter1d(video_array[start:end],
                                                                sigma=sigma,
                                                                axis=0,
                                                                truncate=support/2)[support//2]
        # uncomment to blend in "support" frames, which causes noticeable blurs in some cases
        #video_array[start:end] = gaussian_filter1d(video_array[start:end],
        #                                           sigma=sigma,
        #                                           axis=0,
        #                                           truncate=support/2)
        blurred_video = array_to_images(video_array)
        for i in range(len(video)):
            video[i] = blurred_video[i]
    return video_batch

def calculate_mean_std(image_array, channel):
    channel_data = image_array[:, :, channel]
    return channel_data.mean(), channel_data.std()

def adjust_mean(image, target_mean, channel):
    channel_data = np.array(image)[:, :, channel]
    current_mean = channel_data.mean()
    adjusted_data = channel_data + (target_mean - current_mean)
    adjusted_data = np.clip(adjusted_data, 0, 255).astype(np.uint8)
    image_np = np.array(image)
    image_np[:, :, channel] = adjusted_data
    return Image.fromarray(image_np)

def adjust_contrast(image, target_contrast, channel):
    channel_data = np.array(image)[:, :, channel]
    current_mean = channel_data.mean()
    current_contrast = channel_data.std()
    if current_contrast == 0:
        adjusted_data = current_mean * np.ones_like(channel_data)
    else:
        adjusted_data = (channel_data - current_mean) * (target_contrast / current_contrast) + current_mean
    adjusted_data = np.clip(adjusted_data, 0, 255).astype(np.uint8)
    image_np = np.array(image)
    image_np[:, :, channel] = adjusted_data
    return Image.fromarray(image_np)

def calculate_brightness(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)
    for index in range(scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale

def calculate_contrast(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    mean = sum(i * w for i, w in enumerate(histogram)) / pixels
    contrast = sum((i - mean) ** 2 * w for i, w in enumerate(histogram)) / pixels
    return contrast ** 0.5

def adjust_brightness_contrast(image, target_brightness, target_contrast):
    current_brightness = calculate_brightness(image)

    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(target_brightness / current_brightness)

    current_contrast = calculate_contrast(image)
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(target_contrast / current_contrast)

    return image

def adjust_statistics_to_match_reference(video_batch,
                                         start_index_prediction=17,
                                         reference_window_size=3):
    assert start_index_prediction > 1, f"Need at least 1 frame before prediction start"
    assert start_index_prediction > reference_window_size, f"Reference window size incorrect: {start_index_prediction} <= {reference_window_size}"
    for video in video_batch:

        window_start = max(start_index_prediction - reference_window_size, 0)

        ## first adjust the mean and contrast of each color channel
        #video_array = images_to_array(video)
        #window_frames = video_array[window_start:start_index_prediction]
        #for channel in range(3):
        #    window_mean, window_std = calculate_mean_std(window_frames, channel)
        #    for ii in range(start_index_prediction, len(video)):
        #        video[ii] = adjust_mean(video[ii], window_mean, channel)
        #        video[ii] = adjust_contrast(video[ii], window_std, channel)

        # then adjust the overall brightness and contrast
        window_brightness = np.mean(
            [calculate_brightness(video[jj]) for jj in range(window_start, start_index_prediction)])
        window_contrast = np.mean(
            [calculate_contrast(video[jj]) for jj in range(window_start, start_index_prediction)])
        for ii in range(start_index_prediction, len(video)):
            video[ii] = adjust_brightness_contrast(video[ii], window_brightness, window_contrast)

    return video_batch
def process_video(video_file,output_npz_file):
    res = {}
    # 读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    mask_vf =  VideoReader(video_path.replace('final_ted_v2','final_ted_v2_mask'), ctx=cpu(0))
    bboxes = np.load(video_path.replace('final_ted_v2','final_ted_v2_bbox').replace('mp4','npy'))
    num_frames = len(vr)
    for i in range(num_frames):
        mask = vr[i].asnumpy()/255
        frames = vr[i].asnumpy()*mask
        input_box = bboxes[i]
        x1,y1,x2,y2 = input_box
        frames = frames[int(y1):int(y2),int(x1):int(x2)]
        pdb.set_trace()
        frame_pil = Image.fromarray(frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    # video_files = sorted(glob.glob('/projects/p32321/fating/dataset/CelebV-HQ/35666/*/*.mp4'))
    video_files = sorted(glob.glob(args.input_dir+'/*/*.mp4'))
    # video_files = sorted(glob.glob('/projects/p32296/fating/dataset/HDTF-Processed/*/video.mp4'))
    allocated_fps = [video_files[i] for i in range(len(video_files)) if i % args.parallelism == args.rank]

    pbar = tqdm(total=len(allocated_fps), desc="processing ...")
    for video_file in allocated_fps:
        pbar.update(1)
        # 生成与视频同一目录下的 pose.npz 文件的路径
        output_npz_file = video_file.replace('final_ted_v2','final_ted_v2_brightness').replace('mp4','npy')
        if os.path.exists(output_npz_file):
            continue
        os.makedirs(os.path.dirname(output_npz_file),exist_ok=True)
        try:
            process_video(video_file,output_npz_file)
        except Exception as e:
            print(e)