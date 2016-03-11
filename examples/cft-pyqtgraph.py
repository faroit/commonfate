# -*- coding: utf-8 -*-
"""
quick hack of plotting the 2d-Fourier transform of a STFT slice
"""

import sys
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from commonfate import transform
import argparse
import soundfile as sf


def process_patch(X):
    win = np.outer(
        np.hamming(X.shape[0]), np.hamming(X.shape[1])
    )
    if np.any(np.iscomplex(X)):
        return np.abs(np.fft.fftn(X*win))**0.5
    else:
        return np.abs(np.fft.rfftn(X*win))**0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Modulation Tensors')

    parser.add_argument('input', type=str, help='Input Audio File')

    args = parser.parse_args()

    # STFT parameters
    nfft = 1024
    thop = 256

    # loading signal
    (xwave, fs) = sf.read(args.input)

    print 'computing CFT'
    x_cft = transform.cft(xwave, nfft, thop)

    app = QtGui.QApplication([])

    # Create window with two ImageView widgets
    win = QtGui.QMainWindow()
    win.resize(800, 800)
    win.setWindowTitle('Common Fate Transform Visualizer')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    l = QtGui.QGridLayout()
    cw.setLayout(l)
    imv1 = pg.ImageView()
    imv2 = pg.ImageView()
    imv3 = pg.ImageView()
    l.addWidget(imv1, 0, 0, 1, 2)
    l.addWidget(imv2, 1, 0)
    l.addWidget(imv3, 1, 1)
    win.show()
    roi = pg.RectROI([20, 20], [40, 40], pen='r')
    imv1.addItem(roi)
    textVmax = pg.TextItem(
        "text", anchor=(0.5, 1.5), border='w', fill=(0, 0, 255)
    )
    imv1.addItem(textVmax)
    textVmax.setParentItem(roi)

    data = np.squeeze(x_cft.T)

    def update():
        global data, imv1, imv2
        d2 = roi.getArrayRegion(
            data,
            imv1.imageItem,
            axes=(0, 1),
            returnMappedCoords=True
        )
        x = slice(int(d2[1][0][0, 0]), int(d2[1][0][-1, 0]))
        y = slice(int(d2[1][1][0, 0]), int(d2[1][1][0, -1]))
        textVmax.setText(
            "(%d x %d)" % (x.stop - x.start, y.stop - y.start)
        )

        imv2.setImage(process_patch(data[x, y]), autoLevels=False)
        imv3.setImage(process_patch(abs(data[x, y])), autoLevels=False)

    roi.sigRegionChanged.connect(update)

    # Display the spectrogram
    imv1.setImage(abs(data))

    update()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
