import pylab as plt
import numpy as np
import itertools
import soundfile as sf
import argparse
import seaborn as sns
import commonfate


def displaySTFT(X, name=None):
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    sns.set_style("white")

    fig, ax = plt.subplots(1, 1)
    plt.figure(1)
    plt.pcolormesh(
        np.flipud(abs(np.squeeze(X))),
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',
    )

    if name is None:
        plt.show()
    else:
        plt.savefig(name)


def displayMSTFT(Z, name=None):
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    plt.figure(1)
    (nF, nT) = Z.shape[2:4]
    for (f, t) in itertools.product(range(nF), range(nT)):
        plt.subplot(nF, nT, (nF-f-1) * nT+t+1)
        plt.pcolormesh(
            np.flipud(abs(Z[..., f, t])) ** 0.3,
            vmin=0,
            vmax=10,
            cmap='cubehelix_r',
        )

        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')

    f = plt.gcf()
    f.subplots_adjust(wspace=0, hspace=0)

    if name is None:
        plt.show()
    else:
        plt.savefig(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Common Fate Model')

    parser.add_argument('input', type=str, help='Input Audio File')

    args = parser.parse_args()

    filename = args.input

    # loading signal
    (audio, fs) = sf.read(filename, always_2d=True)

    x_stft = commonfate.transform.forward(audio, 1024, 512)

    x_cft = commonfate.transform.forward(
        x_stft, (64, 32), (32, 16), real=False
    )

    print 'getting modulation spectrogram, shape:', x_cft.shape
    z_cft = np.abs(x_cft)

    displaySTFT(x_stft)
    displayMSTFT(z_cft[..., :, :, 0])
