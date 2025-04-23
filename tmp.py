# -*- coding: utf-8 -*-

import math
import sys
from glob import glob

import librosa
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
from keras import layers
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# 参数
batch_size = 32
epoch = 100
initial_learning_rate = 2e-3
decay_steps = 5000
decay_rate = 0.95
warmup_step = 100000
save_step = 200
feature_num = 80
# 模型结构
pb_path = f'./resources/asr_model'

vocab = []
with open('./resources/vocab_2991.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            vocab.append(line)
vocab.append('unk')
vocab.append('blank_')
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}
char_num = len(vocab)
unk_id = char_num - 2
blank_index = char_num - 1

class AsrModel:
    def __init__(self, voc_num, feature_num):
        self.voc_num = voc_num
        self.audio_mels = layers.Input(shape=[None, feature_num], name='audio_mels')

        asr_output = self.asr_model()

        self.model = tf.keras.Model(inputs=self.audio_mels, outputs=asr_output)
        self.model.summary()

    def asr_model(self):
        input_features = self.audio_mels[:, :, :, tf.newaxis]
        conv_result1 = layers.Conv2D(16, kernel_size=[3, 3], padding='same')(input_features)
        conv_result1 = layers.BatchNormalization(epsilon=1e-5)(conv_result1)
        conv_result1 = layers.ReLU()(conv_result1)
        conv_result1 = layers.MaxPool2D((2, 2))(conv_result1)

        conv_result1 = layers.Conv2D(32, kernel_size=[3, 3], padding='same')(conv_result1)
        conv_result1 = layers.BatchNormalization(epsilon=1e-5)(conv_result1)
        conv_result1 = layers.ReLU()(conv_result1)
        conv_result1 = layers.MaxPool2D((2, 2))(conv_result1)

        conv_result1 = layers.Conv2D(64, kernel_size=[3, 3], padding='same')(conv_result1)
        conv_result1 = layers.BatchNormalization(epsilon=1e-5)(conv_result1)
        conv_result1 = layers.ReLU()(conv_result1)
        conv_result1 = layers.MaxPool2D((2, 2))(conv_result1)

        conv_result2 = layers.Conv2D(16, kernel_size=[5, 5], padding='same')(input_features)
        conv_result2 = layers.BatchNormalization(epsilon=1e-5)(conv_result2)
        conv_result2 = layers.ReLU()(conv_result2)
        conv_result2 = layers.MaxPool2D((2, 2))(conv_result2)

        conv_result2 = layers.Conv2D(32, kernel_size=[5, 5], padding='same')(conv_result2)
        conv_result2 = layers.BatchNormalization(epsilon=1e-5)(conv_result2)
        conv_result2 = layers.ReLU()(conv_result2)
        conv_result2 = layers.MaxPool2D((2, 2))(conv_result2)

        conv_result2 = layers.Conv2D(64, kernel_size=[5, 5], padding='same')(conv_result2)
        conv_result2 = layers.BatchNormalization(epsilon=1e-5)(conv_result2)
        conv_result2 = layers.ReLU()(conv_result2)
        conv_result2 = layers.MaxPool2D((2, 2))(conv_result2)

        conv_result3 = layers.Conv2D(16, kernel_size=[7, 7], padding='same')(input_features)
        conv_result3 = layers.BatchNormalization(epsilon=1e-5)(conv_result3)
        conv_result3 = layers.ReLU()(conv_result3)
        conv_result3 = layers.MaxPool2D((2, 2))(conv_result3)

        conv_result3 = layers.Conv2D(32, kernel_size=[7, 7], padding='same')(conv_result3)
        conv_result3 = layers.BatchNormalization(epsilon=1e-5)(conv_result3)
        conv_result3 = layers.ReLU()(conv_result3)
        conv_result3 = layers.MaxPool2D((2, 2))(conv_result3)

        conv_result3 = layers.Conv2D(64, kernel_size=[7, 7], padding='same')(conv_result3)
        conv_result3 = layers.BatchNormalization(epsilon=1e-5)(conv_result3)
        conv_result3 = layers.ReLU()(conv_result3)
        conv_result3 = layers.MaxPool2D((2, 2))(conv_result3)

        layer_output = conv_result1 + conv_result2 + conv_result3

        layer_output = layers.Reshape([-1, layer_output.shape[2] * layer_output.shape[3]])(layer_output)

        layer_output = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(layer_output)

        layer_output = layers.Dense(512)(layer_output)
        layer_output = layers.ReLU()(layer_output)
        layer_output = layers.Dense(self.voc_num)(layer_output)

        layer_output = layers.Softmax(name='ctc_output')(layer_output)

        return layer_output


class SpeechDataset(Dataset):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data
        self.audio_max_len = 1200
        self.audio_downsample_maxlen = get_audio_len(self.audio_max_len)  # 下采样最大长度
        self.label_max_length = 64

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        wav_path = self.train_data[index][0]
        sentence = self.train_data[index][1]
        sentence = sentence.replace(' ', '')

        label_id = np.array(self.convert_text(sentence))
        label_id = label_id[:self.label_max_length]
        label_len = np.array([len(label_id)])
        label_id = self.pad_zero(label_id, self.label_max_length)

        wav_feature = self.extract_fbank(wav_path)
        wav_feature = wav_feature[:self.audio_max_len]
        wav_feature_len = np.array([get_audio_len(len(wav_feature))])
        wav_feature = self.pad_zero(wav_feature, self.audio_max_len)

        b_data = {'wav_feature': wav_feature, 'wav_feature_len': wav_feature_len,
                  'label_id': label_id, 'label_len': label_len}

        return b_data

    def convert_text(self, text):
        text = text.replace(' ', '')
        ids = []
        for char in text:
            if char in char2idx:
                ids.append(char2idx[char])
            else:
                ids.append(unk_id)
        return ids

    def extract_fbank(self, wav_path, n_mels=80):
        sr = 16000
        hop_length = 160
        win_length = 400
        nfft = 512

        samples = librosa.load(wav_path, sr=sr)[0]

        fbank = librosa.feature.melspectrogram(y=samples, sr=sr, hop_length=hop_length, win_length=win_length,
                                               n_mels=n_mels, n_fft=nfft, htk=True).astype(np.float32)
        fbank = fbank.T
        fbank = librosa.power_to_db(fbank)
        return fbank

    def pad_zero(self, input, length):
        input_shape = input.shape
        if input_shape[0] >= length:
            return input[:length]

        if len(input_shape) == 1:
            return np.append(input, [0] * (length - input_shape[0]), axis=0)

        if len(input_shape) == 2:
            return np.append(input, [[0] * input_shape[1]] * (length - input_shape[0]), axis=0)


def get_audio_len(audio_len):
    audio_len = (audio_len - 2) // 2 + 1
    audio_len = (audio_len - 2) // 2 + 1
    audio_len = (audio_len - 2) // 2 + 1
    return audio_len


def compute_ctc_loss(label_ids, ctc_input, label_length, input_mels_length):
    # blank_index默认-1，无法指定。接口里面说不需要softmax，但貌似无效。需要进行softmax操作，模型已经softmax过。
    loss = tf.keras.backend.ctc_batch_cost(label_ids, ctc_input, input_mels_length.reshape([-1, 1]),
                                           label_length.reshape([-1, 1]))
    # loss = tf.nn.ctc_loss(label_ids, ctc_input, label_length.reshape([-1]), input_mels_length.reshape([-1]),
    #                       logits_time_major=False, blank_index=blank_index)
    loss = tf.reduce_mean(loss)
    return loss


def position_embedding(position, d_model):
    pos_encoding = np.zeros([position, d_model])
    position = np.expand_dims(np.arange(0, position, dtype=np.float32), 1)
    div_term = np.power(10000, -np.arange(0, d_model, 2, dtype=np.float32) / d_model)
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding


def chunks(arr_list, num):
    n = int(math.ceil(len(arr_list) / float(num)))
    return [arr_list[i:i + n] for i in range(0, len(arr_list), n)]


def pre_data():
    train_data = []
    wav_paths = sorted(glob('F:/YYSB/code/code/data1/asr/data_thchs30/data/*.wav'))
    txt_paths = sorted(glob('F:/YYSB/code/code/data1/asr/data_thchs30/data/*.trn'))
    for i in range(len(wav_paths)):
        with open(txt_paths[i], 'r', encoding='utf-8') as f:
            sentence = f.readlines()[0]
            sentence = sentence.replace('\n', '')
            train_data.append([wav_paths[i], sentence])
    return train_data

def load_frame(path):
    frame_names = {}
    for frame_name in glob(f'{path}/frame*'):
        name = os.path.split(frame_name)[1]
        if len(name.split('_')) == 2:
            frame_names[int(name.split('_')[1])] = frame_name

    if len(sorted(frame_names)) == 0:
        return None, None
    else:
        frame_index = sorted(frame_names)[-1]
        return frame_names[frame_index], frame_index

def delete_frame(path):
    frame_names = {}
    for frame_name in glob(f'{path}/frame*'):
        name = os.path.split(frame_name)[1]
        if len(name.split('_')) == 2:
            frame_names[int(name.split('_')[1])] = frame_name

    for delete_key in sorted(frame_names)[:-5]:
        os.remove(frame_names[delete_key])

def greedy_decode(y, blank_index=0):
    def remove_blank(labels, blank=0):
        new_labels = []
        # 合并相同的标签
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # 删除blank
        new_labels = [l for l in new_labels if l != blank]

        return new_labels

    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = np.argmax(y, axis=1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = remove_blank(raw_rs, blank_index)
    return raw_rs, rs


# 编辑距离
def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str(str1[i - 1]) == str(str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]

if __name__ == '__main__':


    ''' ----------------------------加载模型------------------------------- '''
    if not os.path.exists(pb_path):
        os.makedirs(pb_path)

    asr_train_model = None
    asr = AsrModel(char_num, feature_num)
    # 恢复权重
    frame_path, frame_index = load_frame(path=pb_path)
    if frame_path is None:
        global_step = 0
    else:
        asr.model.load_weights(frame_path)
        global_step = frame_index
        print(f'恢复权重 = {frame_path}')
    asr_train_model = asr.model

    ''' ------------------------准备数据------------------------------ '''
    train_data = pre_data()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate))

    for epoch_index in range(epoch):
        loader = DataLoader(dataset=SpeechDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=False,
                            num_workers=4, collate_fn=None)
        for data in tqdm(loader):
            wav_features = np.array(data['wav_feature'])
            wav_feature_lens = np.array(data['wav_feature_len'])
            label_ids = np.array(data['label_id'])
            label_lens = np.array(data['label_len'])
            global_step = global_step + 1

            # 梯度下降
            with tf.GradientTape(persistent=False) as tape:
                asr_output = asr_train_model(wav_features, training=True)

                ctc_loss = compute_ctc_loss(label_ids, asr_output, label_lens, wav_feature_lens)

                grads = tape.gradient(target=ctc_loss, sources=asr_train_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, asr_train_model.trainable_variables))

            print(f'global_step = {global_step}  epoch = {epoch_index + 1}  total_loss = {ctc_loss}')

            label_txt,pre_txt = '',''
            if global_step % 200 == 0:
                cer_result_list = []
                for i in tqdm(range(len(asr_output))):
                    asr_result = asr_output[i][:wav_feature_lens[i][0]]
                    label_id = label_ids[i][:label_lens[i][0]]

                    ctc_beam_out = greedy_decode(asr_result, blank_index=blank_index)
                    cer_result = edit(list(ctc_beam_out[1]), label_id) / len(label_id)

                    label_txt=''.join([idx2char[l] for l in label_id])
                    pre_txt=''.join([idx2char[l] for l in list(ctc_beam_out[1])])
                    cer_result_list.append(cer_result)
                print(np.mean(cer_result_list),label_txt,pre_txt)
                asr_train_model.save(f'{pb_path}/frame_{global_step}', save_format='h5')
                delete_frame(pb_path)
                print(f'保存模型到{pb_path}中')

    asr_train_model.save(f'{pb_path}/frame_{global_step}', save_format='h5')
    delete_frame(pb_path)
    print(f'保存模型，训练完成！')
