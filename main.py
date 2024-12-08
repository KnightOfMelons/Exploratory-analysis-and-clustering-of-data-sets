import pandas as pd
import umap
import trimap
import pacmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score


# ======================================================================================================
# Код сначала загружает данные из файла horse-colic.data, при этом значения ? заменяются на NaN,
# чтобы их можно было корректно обработать. Затем каждому столбцу присваиваются понятные имена,
# чтобы дальше было легче работать с данными. Для числовых переменных, таких как температура, пульс
# и другие, где могут быть пропуски, заполняются недостающие значения медианой по каждому столбцу.
# Для категориальных данных, например, для признаков "хирургия", "возраст" и т. п., пропуски заполняются
# наиболее частым значением, что является стандартной практикой при работе с такими данными. После этого 
# столбец surgical_lesion, удаляется. В конце данные проходят через MinMaxScaler, который масштабирует 
# их в диапазон от 0 до 1, что важно для улучшения работы моделей машинного обучения.
# ======================================================================================================


df = pd.read_csv('WORK_FILES/horse-colic.data', header=None, sep=r'\s+', na_values='?')

df.columns = [
    "surgery", "age", "hospital_number", "rectal_temp", "pulse", "respiratory_rate",
    "extremities_temp", "peripheral_pulse", "mucous_membranes", "capillary_refill_time",
    "pain", "peristalsis", "abdominal_distension", "nasogastric_tube",
    "nasogastric_reflux", "nasogastric_reflux_ph", "rectal_exam_feces", "abdomen",
    "packed_cell_volume", "total_protein", "abdominocentesis_appearance",
    "abdominocentesis_total_protein", "outcome", "surgical_lesion", "lesion_site",
    "lesion_type", "lesion_subtype", "cp_data"
]


linear_columns = ["rectal_temp", "pulse", "respiratory_rate", "nasogastric_reflux_ph",
                  "packed_cell_volume", "total_protein", "abdominocentesis_total_protein"]

for col in linear_columns:
    df[col] = df[col].fillna(df[col].median())

discrete_columns = ["surgery", "age", "extremities_temp", "peripheral_pulse",
                    "mucous_membranes", "capillary_refill_time", "pain", "peristalsis",
                    "abdominal_distension", "nasogastric_tube", "nasogastric_reflux",
                    "rectal_exam_feces", "abdomen", "abdominocentesis_appearance",
                    "outcome"]

for col in discrete_columns:
    df[col] = df[col].fillna(df[col].mode()[0])


data = df.drop(columns = ['surgical_lesion']).to_numpy()

scaled_data = MinMaxScaler().fit_transform(data)

# ======================================================================================================
# Этот код визуализирует результаты кластеризации с помощью четырёх методов пониженной размерности: t-SNE, 
# UMAP, TriMAP и PaCMAP. В функции perform_visualization данные преобразуются с использованием каждого из 
# этих методов и отображаются на графиках для сравнения. Функция visualize используется для визуализации 
# преобразованных данных, выделяя кластеры, шум и центроиды на графиках.
# ======================================================================================================

def perform_visualization(X: np.ndarray, y: np.ndarray, centroids = None) -> None:
    """
    Функция принимает данные X (матрицу признаков) и метки y (кластеры), а также необязательные 
    центроиды. Она применяет методы понижения размерности (t-SNE, UMAP, TriMAP и PaCMAP), чтобы 
    уменьшить размерность данных до 2D и отобразить результаты на четырёх подграфиках.
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Визуализация результатов кластеризации", fontsize=16)

    if centroids is not None:
        X = np.vstack((X, centroids))
        y = np.concatenate((y, [-2] * len(centroids)))
    
    tsne_reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne_reducer.fit_transform(X)
    visualize(X_tsne, y, "t-SNE", ax=axes[0, 0])

    umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    X_umap = umap_reducer.fit_transform(X)
    visualize(X_umap, y, "UMAP", ax=axes[0, 1])

    trimap_reducer = trimap.TRIMAP(n_inliers=10, n_outliers=5, n_random=5)
    X_trimap = trimap_reducer.fit_transform(X)
    visualize(X_trimap, y, "TriMAP", ax=axes[1, 0])

    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_pacmap = pacmap_reducer.fit_transform(X)
    visualize(X_pacmap, y, "PacMAP", ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def visualize(X: np.ndarray, y: np.ndarray, method_name: str, ax=None) -> None:
    """
    Функция визуализирует данные, используя разные цвета для разных кластеров. 
    Она также добавляет легенду, чтобы на графиках было видно, какой цвет соответствует какому кластеру.
    """

    unique_labels = np.unique(y)
    colors = plt.colormaps['tab10']

    for label in unique_labels:
        idx = y == label
        if label == -2:
            ax.scatter(X[idx, 0], X[idx, 1], s=100, c='black', label='Центроиды', zorder=3)
        elif label == -1:
            ax.scatter(X[idx, 0], X[idx, 1], alpha=0.3, c='black', label='Шум', zorder=1)
        else:
            ax.scatter(X[idx, 0], X[idx, 1], alpha=0.7, c=[colors(label)], label=f"Кластер {label}", zorder=1)

    ax.set_title(f"Визуализация методом {method_name}")
    ax.legend()

# ======================================================================================================
# Код и функция ниже предназначены для оценки качества кластеризации, используя три метрики: скорректированный 
# индекс Ренда, метрику однородности и метрику полноты. 

# ДОПОЛНИТЕЛЬНО

# adjusted_rand_score(labels_true, labels_pred). Скорректированный индекс Ренда (ARI) измеряет сходство 
# между двумя кластеризациями, корректируя случайные совпадения. Значение этого показателя варьируется 
# от -1 до 1, где 1 означает полное совпадение, 0 — случайное совпадение, а значения ниже 0 указывают 
# на худшее совпадение, чем случайное.

# homogeneity_score(labels_true, labels_pred). Метрика однородности оценивает, насколько все элементы 
# одного кластера принадлежат к одному и тому же истинному классу. Высокое значение этой метрики означает, 
# что кластеры содержат элементы одного класса.

# completeness_score(labels_true, labels_pred). Метрика полноты измеряет, насколько все элементы одного 
# истинного класса находятся в одном кластере. Высокое значение означает, что каждый истинный класс 
# полностью представлен в одном кластере

# ПРИМЕР РАБОТЫ ИЗ ДАЛЬНЕЙШЕГО КОДА

# Скорректированный индекс Ренда 0.27221489738970717
# Метрика однородности 0.17925461512825452
# Метрика полноты 0.22662298987121685
# ======================================================================================================

def evaluate_clustering(labels_true, labels_pred):
    print(f'\nСкорректированный индекс Ренда {adjusted_rand_score(labels_true, labels_pred)}\n')
    print(f'Метрика однородности {homogeneity_score(labels_true, labels_pred)}\n')
    print(f'Метрика полноты {completeness_score(labels_true, labels_pred)}\n')