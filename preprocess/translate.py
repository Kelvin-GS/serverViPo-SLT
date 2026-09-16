from deep_translator import GoogleTranslator
import numpy as np
import os
from tqdm import tqdm
import time
import random

mode = ["train", "dev", "test"]

for m in mode:
    path = f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/CSL-Daily/{m}_info.npy"

    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True).item()

    languages = {"es": "Spanish", "fr": "French", "en": "English"}

    for i in range(len(data)):
        sentence = data[i]['original_info'].split("|")[-2]
        data[i]['text'] = sentence

        tqdm.write(f"Translating index {i}: {sentence}")
        for lang_code in languages.keys():
            retries = 3
            for attempt in range(retries):
                try:
                    translation = GoogleTranslator(source='auto', target=lang_code).translate(sentence)
                    data[i][f"{lang_code}_text"] = translation
                    tqdm.write(f"  {languages[lang_code]}: {translation}")
                    break
                except Exception as e:
                    if "429" in str(e):
                        sleep_time = random.uniform(5, 15)
                        time.sleep(sleep_time)
                    else:
                        tqdm.write(f"Error translating index {i} to {lang_code}: {e}")
                        data[i][f"{lang_code}_text"] = None
                        break
            time.sleep(random.uniform(0.3, 0.8))


            tqdm.write(f"{i}")

    save_path = path.replace(".npy", "_ml.npy")
    with open(save_path, 'wb') as f:
        np.save(f, data, allow_pickle=True)
