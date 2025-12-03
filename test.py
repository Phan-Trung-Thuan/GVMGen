from module.decoder.models import gvmgen
from module.decoder.data.audio import audio_write
from data_preprocess.utils.video import capture_video
import moviepy.editor as mp
from pydub import AudioSegment
import os
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Script for processing video and model paths.')
    
    parser.add_argument('--model_path', type=str, default='./checkpoints', 
                        help='Path to the model checkpoint.')
    parser.add_argument('--video_path', type=str, required=True, 
                        help='Path to the input video file.')
    parser.add_argument('--syn_path', type=str, required=True, 
                        help='Path to the synthesis output directory.')
    parser.add_argument('--fps', type=int, default=1, 
                        help='video sample rate.')
    parser.add_argument('--duration', type=int, default=30, 
                        help='video length.')
    
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = gvmgen.GVMGen.get_pretrained(args.model_path, device=device)

    mp4_pt = capture_video(args.video_path, args.fps, device, args.duration)
    model.set_generation_params(duration=mp4_pt.shape[0])

    description = [mp4_pt]

    res = model.generate(descriptions = description)

    for idx, one_wav in enumerate(res):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        video_mp = mp.VideoFileClip(str(args.video_path))
        audio_clip = AudioSegment.from_wav(str(idx)+'.wav')
        audio_clip[0:int(video_mp.duration*1000)].export(str(idx)+'.wav')
        # Render generated music into input video
        audio_mp = mp.AudioFileClip(str(str(idx)+'.wav'))

        audio_mp = audio_mp.subclip(0, video_mp.duration )
        final = video_mp.set_audio(audio_mp)
        try:
            final.write_videofile(os.path.join(args.syn_path, str(idx)+'.mp4'),
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
        except Exception as e:
            print(f"error：{e}")
        os.remove(str(idx)+'.wav')

if __name__ == '__main__':
    main()

