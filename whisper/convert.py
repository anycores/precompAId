from onnx import numpy_helper
import numpy as np
import onnx
import ffmpeg
import argparse

# Parameter settings
parser = argparse.ArgumentParser(description='Whisper format converter')
parser.add_argument('--ipath', metavar='S', help='path to the input file')
parser.add_argument('--opath', metavar='S', help='path to the output file (.pb extension)')
args = parser.parse_args()

if __name__ == '__main__':

    out, _ = (
        ffmpeg.input(args.ipath, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    onnx_tp = numpy_helper.from_array(audio, 'raw_audio')
    onnx.save_tensor(onnx_tp, args.opath)
