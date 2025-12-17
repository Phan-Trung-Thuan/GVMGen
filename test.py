from module.decoder.models import gvmgen
from module.decoder.data.audio import audio_write
from data_preprocess.utils.video import capture_video
import moviepy.editor as mp
from pydub import AudioSegment
import os
import torch
import argparse

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
def main():
    peak_mem_0 = 0
    peak_mem_1 = 0

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
    print(count_parameters(model))

    description = [mp4_pt]

    import time
    from tqdm import tqdm
    start = time.time()
    res = model.generate(descriptions = description)

    for idx, one_wav in tqdm(enumerate(res)):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        video_mp = mp.VideoFileClip(str(args.video_path))
        video_ms = int(video_mp.duration * 1000)

        audio_clip = AudioSegment.from_wav(str(idx)+'.wav')

        # Pad or trim precisely to match video duration
        if len(audio_clip) < video_ms:
            # Add silence if audio is shorter
            silence = AudioSegment.silent(duration=video_ms - len(audio_clip))
            audio_clip = audio_clip + silence
        else:
            # Trim if too long
            audio_clip = audio_clip[:video_ms]

        audio_clip.export(str(idx)+'.wav')
        
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

        peak_mem_0 = max(peak_mem_0, torch.cuda.max_memory_allocated(0) / 1024 / 1024)
        peak_mem_1 = max(peak_mem_1, torch.cuda.max_memory_allocated(1) / 1024 / 1024)

    end = time.time()
    print(f'Processing time: {end - start}')

    print(f'Memory usage: {peak_mem_0:.2f} MB')
    print(f'Memory usage: {peak_mem_1:.2f} MB')

if __name__ == '__main__':
    main()

