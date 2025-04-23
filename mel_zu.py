import librosa
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('TkAgg')

sr = 16000
mel_basis = librosa.filters.mel(sr=sr, n_fft=512, n_mels=10,fmin=0, fmax=sr / 2)
plt.plot(mel_basis)
plt.show()
plt.plot(mel_basis.T)
plt.show()

sr = 16000
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

mel_basis = librosa.filters.mel(sr=sr, n_fft=512, n_mels=30,fmin=0, fmax=sr / 2)
mel_basis /= np.max(mel_basis, axis=-1)[:, None]

plt.plot(mel_basis)
plt.show()

#front是标签属性：包括字体、大小等
font = {
'weight' : 'normal',
'size'   : 11,
}
plt.xlabel("Episode",font)
plt.ylabel(r"Average Reward",font)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0,300)
plt.ylim(0,1)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude' )
plt.plot(mel_basis.T)
plt.show()

