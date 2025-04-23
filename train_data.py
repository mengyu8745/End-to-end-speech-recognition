from utils import gendict
from model_list import CNN_LSTM_CTC, CLDNN_CTC, MRDCNN_CTC, MR_CLDNN_CTC, CNN_LSTM, CLDNN,MRDCNN,MR_CLDNN
from utils import data_generate, data_generate_1
import matplotlib.pyplot as plt
# 训练模型
# 准备训练所需数据
def train(train_data_path, textfile, bath_size, model_save_path, model_data_save_path, model_name):
    CTC_model_name_list = ['CNN_LSTM_CTC','CLDNN_CTC','MRDCNN_CTC','MR_CLDNN_CTC']
    CTC_mode_list = {
        'CNN_LSTM_CTC': CNN_LSTM_CTC(),
        'CLDNN_CTC': CLDNN_CTC(),
        'MRDCNN_CTC': MRDCNN_CTC(),
        'MR_CLDNN_CTC': MR_CLDNN_CTC()
    }
    not_CTC_model_name_list = ['CNN_LSTM','CLDNN', 'MRDCNN', 'MR_CLDNN']
    not_CTC_mode_list = {
        'CNN_LSTM': CNN_LSTM(),
        'CLDNN': CLDNN(),
        'MRDCNN': MRDCNN(),
        'MR_CLDNN': MR_CLDNN()
    }
    # 导入模型结构，训练模型，保存模型参数
    if model_name in CTC_model_name_list:
        model, model_data = CTC_mode_list[model_name]
        yielddatas = data_generate(train_data_path, textfile, bath_size)
        history = model.fit_generator(yielddatas, steps_per_epoch=1000, epochs=100)
        model.save(model_save_path)
        model_data.save(model_data_save_path)
        print(history.history)


    if model_name in not_CTC_model_name_list:
        model = not_CTC_mode_list[model_name]
        X, y = data_generate_1(train_data_path, textfile, bath_size)
        history = model.fit(X, y, epochs=100, batch_size=32)
        model.save(model_save_path)
        print(history.history)
