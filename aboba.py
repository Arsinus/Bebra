import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from pennylane import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from test_qugan import *
import base64


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("aboba/maxresdefault_live.jpg")

st.title("MedMNIST")

st.header("Датасет, PCA")

st.markdown("Датасет состоит из трёх наборов рентгенов: черепа, руки и грудной клетки.")
st.markdown("Для визуализации используется метод PCA (Principal component analysis), который сжимает датасет до нужной размерности и при этом сохраняет взаимосвязи между его элементами")
st.markdown("Сжатый датасет находится в латентном пространстве (в данном случае двумерном)")


with st.container():
    coord = np.array([0,0])
    st.subheader("Латентное пространство")
    WIDTH_1 = 300
    col1, col2 = st.columns([0.5,0.5])
    with col1:
        value = streamlit_image_coordinates("aboba/trans_pca.png", width=WIDTH_1, height=WIDTH_1)
    try:
        coord = (np.array([value['x'], WIDTH_1 - value['y']]) - np.array([0.1625*WIDTH_1, 0.145*WIDTH_1])) / np.array([0.7*WIDTH_1])
    except:
        print()
    img = pca.inverse_transform(scale.inverse_transform(coord.reshape(1, -1))).reshape(64, 64)
    fig, ax = plt.subplots()
    plt.imshow(img, origin='lower', cmap='gray')
    plt.axis('off')
    plt.savefig('aboba/dec.png', bbox_inches="tight")
    plt.clf()
    col2.image('aboba/dec.png', width=WIDTH_1)

with st.container():
    st.subheader("PCA энкодер-декодер")

    st.markdown("При сжатии PCA мы теряем часть информации, но эта потеря уменьшается засчёт того, что PCA учитывает взаимное расположение между элементами датасета")
    st.markdown("Изначальная размерность элементов датасета (изображений) $64\cdot64=4096$, мы сжимаем её до $2$, чтобы визуализировать на плоскости, но по своей сути размерностью модель не ограничена")

    WIDTH_2 = 200
    test_dim = st.slider("Размерность латентного пространства (Мы используем 2 для визуализации)", 1, 600, step=1)
    col1, col2, col3 = st.columns(3)
    col1.image("MedMnist/CXR/000004.jpeg", width=WIDTH_2)
    plt.imshow(pca_ende("MedMnist/CXR", "000004.jpeg", test_dim), origin='lower', cmap='gray')
    plt.axis('off')
    plt.savefig('aboba/ende.png', bbox_inches="tight")
    plt.clf()
    col1.image('aboba/ende.png')

    col2.image("MedMnist/HeadCT/000000.jpeg", width=WIDTH_2)
    plt.imshow(pca_ende("MedMnist/HeadCT", "000000.jpeg", test_dim), origin='lower', cmap='gray')
    plt.axis('off')
    plt.savefig('aboba/ende.png', bbox_inches="tight")
    plt.clf()
    col2.image('aboba/ende.png')

    col3.image("MedMnist/Hand/000010.jpeg", width=WIDTH_2)
    plt.imshow(pca_ende("MedMnist/Hand", "000010.jpeg", test_dim), origin='lower', cmap='gray')
    plt.axis('off')
    plt.savefig('aboba/ende.png', bbox_inches="tight")
    plt.clf()
    col3.image('aboba/ende.png')

st.header("GAN")

with st.container():

    st.markdown("GAN (Generative Adversial Network) - Это модель, состоящая из двух нейросетей: дискриминатор и генератор. Цель дискриминатора – научиться различать сгенерированные данные от настоящих, а генератора – создавать такие образцы, которые дискриминатор не сможет отличить от настоящих")

    st.subheader("Обучение")

    st.markdown("На видео показана эволюция модели:")
    st.markdown("- красные точки - реальные данные")
    st.markdown("- красное распределение - сгенерированные данные")
    st.markdown("- жёлтым показано то, что дискриминатор считает реальными данными, синим - нереальные")
    st.markdown("Дискриминатор и генератор делают шаги поочерёдно. Дискриминатор пытается занять область с реальными данными, а генератор стремится в центр области дискриминатора")
    st.markdown("Мы добавили в модель \"Личинусов\". Это метод, который разделяет датасет на кластеры и обучает модель отдельно для каждого кластера, сохраняя веса при минимуме ошибки")
    st.video("Видосы/vay_kakoy.mp4")
    col1, col2 = st.columns(2)
    WIDTH_3 = 300
    n_sc = NOISE_SCALE
    def draw_sample(n_sc):
        sample = get_samples(1, disc_weights, gen_weights, n_sc)
        img = pca.inverse_transform(scale.inverse_transform(sample.reshape(1, -1))).reshape(64, 64)
        plt.imshow(img, origin='lower', cmap='gray')
        plt.axis('off')
        plt.savefig('aboba/sample.png', bbox_inches="tight")
        plt.clf()
        col2.image('aboba/sample.png', width=WIDTH_3)
    files = os.listdir('weights')
    print(files)
    index = st.slider("Weights", 0, len(files) - 1)
    disc_weights, gen_weights = load_weights("weights/model-{}-{}-{}.txt".format(index // BATCH_SIZE, 0, index % BATCH_SIZE))
    n_sc = col1.slider("Noise:", 0., 0.5, NOISE_SCALE)
    save_lanent_space_foraboba(n_sc, disc_weights, gen_weights, latent_clusters[index // BATCH_SIZE], image_size, index // BATCH_SIZE, 100)
    col1.image("aboba/show_progress.png", width=WIDTH_3)


    col1.button("Gen Sample", on_click=draw_sample(n_sc))

st.subheader("QuGAN")

st.markdown("QuGAN это квантовый аналог GAN, который превосходит его по многим аспектам:")
st.markdown("- число параметров снижено на **95%**")
st.markdown("- при том же числе параметров, производительность выше на **48%**")
st.markdown("- меньшая склонность к переобучению (сгенерированное распределение лучше описывает реальное)")

st.subheader("Структура QuGAN")
st.markdown("QuGAN обучается в три этапа:")
st.markdown("- **Обучение дискриминатора на реальных данных**. ")
st.markdown("Закодировать классические данные в квантовое состояние можно двумя способами. "
            "Первый - это закодировать каждую координату в отдельный кубит. В этом случае не используется квантовое преимущество при кодировании данных. "
            "Второй - использовать **амплитудное кодирование**, это позволяет кодировать классические данные, используя $log_2(DIM)$ кубитов. "
            "Так как мы работаем с двумерным пространством необходимости во втором способе нет и мы используем первый с двумя кубитами. ")
st.markdown("Мы получаем значение блока Swap Test. Это блок, который сравнивает два квантовых состояния и возвращает **0** для ортогональных состояний и **1** для коллинеарных. "
            "Это можно интерпретировать так, что мы получаем вероятность того, что дискриминатор считает данное квантовое состояние реальным.")
st.markdown("На данном этапе мы максимизируем это значение, изменяя веса **дискриминатора**")
st.image("aboba/train_on_real.jpg")
st.markdown("- Обучение дискриминатора на сгенерированных данных")
st.markdown("В улучшенной версии мы используем расширенную структуру для генератора и дискриминатора. Были добавлены $N_{anc}$ ancilla кубитов, это позволило генерировать более разнообразные распределения. "
            "Swap Test используется для $N_{dim}$ кубитов, в которые кодируются случайные точки из латентного пространства. "
            "На данном этапе мы минимизируем это значение, изменяя веса **дискриминатора**")
st.markdown("- Обучение генератора")
st.markdown("Аналогично предыдущему этапу, но теперь мы максимизируем Swap Test, изменяя веса **генератора**")
st.image("aboba/gen_sample.jpg", "Цепь для генерации")
st.image("aboba/train_on_fake.jpg", "Цепь для обучения")

st.subheader("Конструктор квантовой цепи")

circ_type = st.radio(
    "Тип цепи",
    ('Train on real', 'Train on fake', 'Gen sample'))
circ_Na = st.slider("Число анцилл", 0, 3, N_a, 1)
circ_Ndim = st.slider("Размерность", 0, 3, N_dim, 1)
circ_depth = st.slider("Глубина", 0, 4, depth, 1)

def draw_circ():
    if circ_type == "Train on real":
        fig, ax = qml.draw_mpl(trn_rl)(circ_Na, circ_Ndim, circ_depth)
        fig.savefig("aboba/construct")
        # plt.clf()
    if circ_type == "Train on fake":
        fig, ax = qml.draw_mpl(trn_fk)(circ_Na, circ_Ndim, circ_depth)
        fig.savefig("aboba/construct")
        # plt.clf()
    if circ_type == "Gen sample":
        fig, ax = qml.draw_mpl(gn_smpl)(circ_Na, circ_Ndim, circ_depth)
        fig.savefig("aboba/construct")
        # plt.clf()

st.button("Собрать цепь", on_click=draw_circ)

st.image("aboba/construct.png")



st.header("Сравнение с оригиналом")
st.markdown("Для оценки качества модели используется расстояние Хеллингера. Это метрика для распределений, которая возвращает 1, если распределения не пересекаются и 0, если они идентичны. ")
st.image("aboba/comparison.jpg")
st.markdown("Сравнение делалось на датасете MNIST для цифр **3**, **6** и **9** с использованием PCA до размерности **2**")
st.markdown("Наша модель обучилась до приемлемого результата за одну эпоху (HD = 0.35). За 20 эпох наша модель достигла результата, которого оригинал достиг за 35 эпох. "
            "Расстояние Хеллингера 0.4 означает, что сгенерированное распределение попало на реальный кластер, дальше нейросеть адаптируется под форму кластера. "
            "Примерно того же результата можно было бы добиться быстрее, если на 5-6 эпохе подобрать подходящий шум, определяющий размер сгенерированного распределения")

print(coord)
