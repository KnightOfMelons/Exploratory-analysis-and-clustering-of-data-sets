import pandas as pd
import umap
import trimap
import pacmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster



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

# labels_surgical_lesion = df['surgical_lesion'].to_numpy()
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

# =========================== Иерархическая алгомеративная кластеризация ===============================
# Код выполняет иерархическую кластеризацию с использованием метода Уорда. Функция linkage строит матрицу 
# расстояний между кластерами, используя данные scaled_data, которые были предварительно масштабированы. 
# В качестве метрики расстояния используется евклидово расстояние, а метод агломерации — метод Уорда, 
# который минимизирует внутрикластерное рассеяние. Результат работы функции — матрица, где каждая строка 
# содержит информацию о двух объединённых кластерах: их индексы, расстояние между ними и количество объектов 
# в новом кластере. 
# ======================================================================================================

distance_matrix = linkage(scaled_data, method='ward', metric='euclidean')

# Можно вывести так, но результат особо понятен не будет.
# print(distance_matrix)

# Будет что-то типа: 

#array([[2.60000000e+01, 2.33000000e+02, 4.86499635e-05, 2.00000000e+00],
#       [1.75000000e+02, 2.88000000e+02, 2.06715236e-03, 2.00000000e+00],
#       [7.30000000e+01, 2.35000000e+02, 2.28654829e-03, 2.00000000e+00],
#       ...,
#       [5.93000000e+02, 5.95000000e+02, 9.42958023e+00, 1.23000000e+02],
#       [5.94000000e+02, 5.96000000e+02, 1.08393604e+01, 2.36000000e+02],
#       [5.92000000e+02, 5.97000000e+02, 1.14473478e+01, 3.00000000e+02]])

df_good_looking = pd.DataFrame(distance_matrix, columns=['Cluster 1', 'Cluster 2', 'Distance', 'Size'])
print(df_good_looking)

# ======================================================================================================
# Код строит дендрограмму для визуализации результатов иерархической кластеризации, используя матрицу 
# расстояний distance_matrix, полученную ранее с помощью функции linkage. Для построения дендрограммы 
# используется функция dendrogram из библиотеки scipy.cluster.hierarchy, а для отображения графика — 
# библиотека matplotlib.

# СМОТРИ EXAMPLES/Dendogram_distance_matrix.png

# ======================================================================================================

fig = plt.figure(figsize=(15, 30))
fig.patch.set_facecolor('white')

R = dendrogram(distance_matrix,
               labels = df['surgical_lesion'].to_numpy(),
               orientation='left',
               leaf_font_size=12)

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код выполняет иерархическую кластеризацию с помощью ранее вычисленной матрицы расстояний и визуализирует 
# результаты, сравнивая реальные метки классов с результатами кластеризации.

# СМОТРИ EXAMPLES/Comparison_of_real_class_labels_with_clustering_results.png

# ======================================================================================================

cluster_labels = fcluster(distance_matrix, 2, criterion='maxclust')
known_labels = df['surgical_lesion'].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].scatter(scaled_data[:, 18], scaled_data[:,19], c=known_labels, cmap=plt.cm.Set1)
axes[1].scatter(scaled_data[:, 18], scaled_data[:,19], c=cluster_labels, cmap=plt.cm.Set2)
axes[0].set_title(f'Реальные метки кластеров', fontsize=16)
axes[1].set_title(f'Метки кластеров, найденные алгоритмом', fontsize=16)

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код сначала оценивает качество кластеризации, сравнив истинные метки классов (known_labels) с метками 
# кластеров, присвоенными алгоритмом (cluster_labels), с помощью трех метрик: скорректированный индекс 
# Ренда, однородность и полнота. Затем он визуализирует результаты кластеризации, используя методы 
# понижения размерности (t-SNE, UMAP, TriMAP и PaCMAP), чтобы отобразить данные в 2D и показать, как 
# алгоритм разделил объекты на кластеры.

# СМОТРИ EXAMPLES/Visualization_of_clustering_results.png

# ======================================================================================================

evaluate_clustering(known_labels, cluster_labels)
perform_visualization(scaled_data, cluster_labels)

# =================================== Итерационные алгоритмы. KMeans ===================================

# Код применяет алгоритм KMeans для кластеризации данных. Алгоритм ищет два кластера, используя инициализацию 
# 'k-means++' и выполняет до 300 итераций с 10 различными начальными точками для повышения стабильности 
# результата. Затем он визуализирует результаты: на первом графике отображаются реальные метки кластеров, 
# а на втором — метки, полученные методом KMeans. В обоих графиках на соответствующих точках отображаются 
# центроиды кластеров (черные точки).

# СМОТРИ EXAMPLES/KMeans_for_data_clustering.png

# ======================================================================================================

warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak on Windows.*")

kmeans = KMeans(n_clusters = 2, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)

cluster_labels = kmeans.fit_predict(scaled_data)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].scatter(scaled_data[:, 18], scaled_data[:, 19], c=known_labels)
axes[0].scatter(kmeans.cluster_centers_[:, 18], kmeans.cluster_centers_[:, 19],
                s = 100, c = 'black', label = 'Центроиды')
axes[0].set_title(f'Реальные метки кластеров', fontsize=16)

axes[1].scatter(scaled_data[:, 18], scaled_data[:,19], c=cluster_labels, cmap=plt.cm.Set2)
axes[1].scatter(kmeans.cluster_centers_[:, 18], kmeans.cluster_centers_[:, 19],
                s = 100, c = 'black', label = 'Центроиды')
axes[1].set_title(f'Метки кластеров, найденные алгоритмом', fontsize=16)

plt.tight_layout()
plt.show()

# ======================================================================================================