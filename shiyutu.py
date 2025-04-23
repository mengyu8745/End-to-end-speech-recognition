# 一个时域图
import matplotlib
matplotlib.use('TkAgg')  # Or you can use 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt
import numpy as np
import wave

# 设置全局的字体大小和英文字体
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix'  # 对于数学公式也使用Times New Roman
})

# # 对于中文标签，使用FontProperties指定字体
# from matplotlib.font_manager import FontProperties
# chinese_font = FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")  # 指向系统的宋体



# 路径根据您的文件实际位置替换
# file_path = r'F:\YYSB\code\code\data1\asr\data_thchs30\test\D4_751.wav'
file_path = r'F:\YYSB\code\code\data1\asr\data_thchs30\test-noise\0db\cafe\D4_751.wav'
# 打开WAV文档
with wave.open(file_path, 'rb') as wav_file:
    # 获取WAV文件的全部帧数
    nframes = wav_file.getnframes()
    # 读取全部的帧
    frames = wav_file.readframes(nframes)
    # 将读取的帧转为二进制的格式
    samples = np.frombuffer(frames, dtype=np.int16)

# 创建时间轴数据点
duration = nframes / wav_file.getframerate()
times = np.linspace(0, duration, num=nframes)

# 绘制时域图
plt.figure(figsize=(18, 5))
plt.plot(times, samples)
# plt.title('Time-domain Signal')
plt.xlabel('Time /s')
plt.ylabel('Amplitude')

# 设置坐标轴范围
plt.xlim(0, duration)  # 横坐标从0开始到语音结尾
plt.ylim(-30000,30000,10000 )  # 纵坐标从-30000开始到samples的最大值
# plt.ylim(min(samples), max(samples))  # 纵坐标从-30000开始到samples的最大值

# 添加网格
plt.grid(True, linestyle='--', color='lightgrey')

# 自动调整布局以去掉多余的白边
plt.tight_layout()

plt.show()
# 两个时域图
# import matplotlib
# matplotlib.use('TkAgg')  # Or you can use 'Qt5Agg', 'GTK3Agg', etc.
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# # 设置支持中文的字体，这里以微软雅黑为例
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#
# import numpy as np
# import wave
#
# # 第一张图（干净语音信号）的路径
# clean_file_path = r'F:\YYSB\code\code\data1\asr\data_thchs30\test\D4_751.wav'
#
# # 第二张图（噪声信号）的路径
# noise_file_path = r'F:\YYSB\code\code\data1\asr\data_thchs30\test-noise\0db\cafe\D4_751.wav'
#
# # 创建一个绘图窗口
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
#
# # 绘制干净语音信号的时域图
# with wave.open(clean_file_path, 'rb') as wav_file:
#     nframes = wav_file.getnframes()
#     frames = wav_file.readframes(nframes)
#     samples = np.frombuffer(frames, dtype=np.int16)
#     duration = nframes / wav_file.getframerate()
#     times = np.linspace(0, duration, num=nframes)
#     ax2.plot(times, samples)
#     ax2.set_title('干净语音信号')
#     ax2.set_xlabel('Time [s]')
#
# # 绘制噪声信号的时域图
# with wave.open(noise_file_path, 'rb') as wav_file:
#     nframes = wav_file.getnframes()
#     frames = wav_file.readframes(nframes)
#     samples = np.frombuffer(frames, dtype=np.int16)
#     duration = nframes / wav_file.getframerate()
#     times = np.linspace(0, duration, num=nframes)
#     ax1.plot(times, samples)
#     ax1.set_title('噪声信号')
#     ax1.set_xlabel('Time [s]')
#     ax1.set_ylabel('Amplitude')
#
# # 显示整个画布
# plt.tight_layout()
# plt.show()

