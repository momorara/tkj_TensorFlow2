# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""
"""
データをリサイズして保存

推奨サイズ：128×128（Raspberry Piでも高速処理可能）

80%を学習用、20%を検証用に分割

出来たデータの数を確認
ls dataset/train/Cat | wc -l
ls dataset/train/Dog | wc -l
ls dataset/val/Cat | wc -l
ls dataset/val/Dog | wc -l
"""
# -*- coding: utf-8 -*-
# -----------------------------------------
# dataset_s → dataset_sr 画像リサイズコピー
# -----------------------------------------
import os
import shutil
from PIL import Image

# 元のデータセット
src_base = 'dataset_s'
# 出力先データセット
dst_base = 'dataset_sr'

# リサイズ後の画像サイズ
IMG_SIZE = (128, 128)

# クラスと分割
classes = ['Cat', 'Dog']
splits  = ['train', 'val']

# ====== 出力フォルダ削除（再作成） ======
if os.path.exists(dst_base):
    shutil.rmtree(dst_base)
    print(f"削除しました: {dst_base}")

# ====== 出力フォルダ作成 ======
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(dst_base, split, cls), exist_ok=True)

print(f"作成しました: {dst_base}")

# ====== リサイズ処理 ======
for split in splits:
    for cls in classes:
        src_dir = os.path.join(src_base, split, cls)
        dst_dir = os.path.join(dst_base, split, cls)

        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"処理中: {split}/{cls} ")

        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            try:
                # 画像を開く
                with Image.open(src_path) as img:
                    # RGBに変換（白黒やRGBAを統一）
                    img = img.convert('RGB')
                    # リサイズ
                    img = img.resize(IMG_SIZE)
                    # 保存（JPEG形式）
                    img.save(dst_path, format='JPEG', quality=95)
            except Exception as e:
                # print(f"⚠️ エラー: {src_path} → {e}")
                pass

print("すべての画像をリサイズして保存しました。")
