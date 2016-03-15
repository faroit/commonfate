import os
import numpy as np
import soundfile as sf
import argparse
from commonfate import decompose


def export(input, input_file, output_path, samplerate):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    basepath = os.path.join(
        output_path, os.path.splitext(os.path.basename(input_file))[0]
    )

    # Write out all components
    for i in range(input.shape[0]):
        sf.write(
            basepath + "_cpnt-" + str(i) + ".wav",
            input[i],
            samplerate
        )

    out_sum = np.sum(input, axis=0)
    sf.write(basepath + '_reconstruction.wav', out_sum, samplerate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Common Fate Model')

    parser.add_argument('input', type=str, help='Input Audio File')

    args = parser.parse_args()

    filename = args.input

    # loading signal
    (audio, fs) = sf.read(filename, always_2d=True)

    out = decompose.process(
        audio,
        nb_iter=10,
        nb_components=2,
        n_fft=1024,
        n_hop=256,
        cft_patch=(32, 48),
        cft_hop=(16, 24)
    )

    export(out, filename, 'output', fs)
