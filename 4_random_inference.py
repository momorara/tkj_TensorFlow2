# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""
# -----------------------------------------
# ランダム画像を選んで推論し、
# スペースで次の画像、qで終了
# -----------------------------------------
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 学習済みモデルの読み込み
model = load_model('cats_vs_dogs_cnn.h5')

# 推論対象フォルダ（猫・犬の両方を含む上位フォルダ）
base_dir = 'dataset_s/val'

# 猫・犬フォルダをリスト化
categories = ['Cat', 'Dog']
img_list = []

for category in categories:
    folder_path = os.path.join(base_dir, category)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_list.extend(files)

if not img_list:
    print("画像ファイルが見つかりません。フォルダのパスを確認してください。")
    exit()

# 隠しメタデータファイルなどを削除
img_list = [f for f in img_list if "/._" not in f]


# -----------------------------------------
# matplotlib ウィンドウを作成
# -----------------------------------------
fig, ax = plt.subplots()
plt.axis('off')  # 軸を非表示

# グローバル変数で表示用画像
img_show = None

def show_random_image(event=None):
    """ランダムに画像を選んで推論＆表示"""
    global img_show
    img_path = random.choice(img_list)
    print(img_path)

    # 推論用にサイズ変更
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]
    if pred < 0.5:
        result = f"result: cat={pred:.3f}"
    else:
        result = f"result: dog={pred:.3f}"

    print("選ばれた画像:", img_path)
    print(result)

    # 表示用画像（元サイズ）
    img_show = image.load_img(img_path)
    ax.clear()
    ax.imshow(img_show)
    ax.axis('off')
    # ax.set_title(result, fontproperties="IPAPGothic")  # 日本語対応
    ax.set_title(result)  
    fig.canvas.draw()

def on_key(event):
    """キー入力イベント"""
    if event.key == ' ':
        try:
            show_random_image()
        except:
            pass
    elif event.key == 'q':
        plt.close(fig)

# キーイベントを接続
fig.canvas.mpl_connect('key_press_event', on_key)

# 最初の画像を表示
show_random_image()

plt.show()
