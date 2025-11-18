# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""
"""
犬猫分類 推論サンプル（全画像順次処理）
- 事前に学習済みモデル cats_vs_dogs_cnn.h5 が必要
- valフォルダ内の全画像に対して推論し、結果を表示
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import shutil
import glob

# 学習済みモデルを読み込み
model = load_model('cats_vs_dogs_cnn.h5')

# valフォルダのパス
val_dir = 'dataset_s/val'
classes = ['Cat', 'Dog']
# エラー画像保存フォルダ
err_dir = 'result_err'
# -----------------------------------------
# result_errを削除して新規作成
# -----------------------------------------
if os.path.exists(err_dir):
    shutil.rmtree(err_dir)
os.makedirs(err_dir, exist_ok=True)
# 判定エラーした画像をerr_dirに保存する:1 しない:0
err_save = 1

dog_n,dogdog,dogcat = 0,0,0
cat_n,catdog,catcat = 0,0,0

print()
# クラスごとに画像を取得して順次推論
for cls in classes:
    cls_dir = os.path.join(val_dir, cls)
    print(cls,'start')
    for fname in os.listdir(cls_dir):
        # print(fname)
        if "._" not in fname :
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')) :
                try:
                    img_path = os.path.join(cls_dir, fname)
                    # 画像読み込み・リサイズ・正規化
                    img = image.load_img(img_path, target_size=(128,128))
                    x = image.img_to_array(img) / 255.0
                    x = np.expand_dims(x, axis=0)
                    
                    # 推論
                    pred = model.predict(x, verbose=0)[0][0]
                    if pred < 0.5:
                        predicted_class = '猫'
                    else:
                        predicted_class = '犬'

                    # print(cls_dir,predicted_class)
                    # break

                    if cls_dir == "dataset_s/val/Cat":
                        cat_n = cat_n +1
                        if predicted_class == '猫':
                            catcat = catcat +1
                        else:
                            # 間違った
                            catdog = catdog +1
                            if err_save == 1:
                                # エラー画像をresult_errに保存
                                shutil.copy(img_path, err_dir)
                    else:
                        dog_n = dog_n +1
                        if predicted_class == '犬':
                            dogdog = dogdog +1
                        else:
                            # 間違った
                            dogcat = dogcat +1
                            if err_save == 1:
                                # エラー画像をresult_errに保存
                                shutil.copy(img_path, err_dir)
                    
                    # 結果をプリント
                    # print(f'{cls} {fname} → 推定: {predicted_class} (確率={pred:.2f})')
                    if dog_n % 50 == 0 and cls == 'Dog':
                        print('.',end='', flush=True)
                    if cat_n % 50 == 0 and cls == 'Cat':
                        print('.',end='', flush=True)
                except:
                    print('エラー:',img_path)
    print()
    print(cls,'end')
print('dog:',dog_n,dogdog,dogcat,int(dogdog/dog_n*1000)/10,'%')
print('cat:',cat_n,catcat,catdog,int(catcat/cat_n*1000)/10,'%')
