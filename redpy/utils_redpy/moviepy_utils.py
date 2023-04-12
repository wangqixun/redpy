from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_audioclips, AudioClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, transfx, vfx, VideoClip, ImageSequenceClip
from moviepy.editor import TextClip, ImageClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy import editor
import os

__all__ = [
    'VideoFileClip', 'ImageSequenceClip', 'CompositeVideoClip', 'concatenate_videoclips',
    'write_video_file'
]


def write_video_file(v_clip, out_path, fps=None, threads=None):
    tmp_audio = os.path.basename(out_path)+'.aac'
    try:
        v_clip.write_videofile(out_path, fps=fps, codec='libx264', audio_codec='aac',
                               temp_audiofile=f'/tmp/{tmp_audio}',
                               threads=threads)
    except IndexError:
        # Short by one frame, so get rid on the last frame:
        v_clip = v_clip.subclip(t_end=(v_clip.duration - 1.0/v_clip.fps))
        v_clip.write_videofile(out_path, fps=fps, codec='libx264', audio_codec='aac',
                               temp_audiofile=f'/tmp/{tmp_audio}',
                               threads=threads)
        print("Saved .mp4 after Exception at {}".format(out_path))
    except Exception as e:
        print("Exception {} was raised!!".format(e))
    return out_path
