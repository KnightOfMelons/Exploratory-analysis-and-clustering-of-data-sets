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
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from fcmeans import FCM



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

# =================================== Итерационные алгоритмы ===========================================
# KMeans.

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
# Делает всё абсолютно то же, но с KMeans

# СМОТРЕТЬ EXAMPLES/KMeans_visualization_results_clustering.png

# ======================================================================================================

evaluate_clustering(known_labels, cluster_labels)
perform_visualization(scaled_data, cluster_labels, kmeans.cluster_centers_)

# ======================================================================================================
# FCM

# Код использует алгоритм Fuzzy C-Means (FCM) для кластеризации данных. В отличие от KMeans, FCM выполняет 
# мягкую кластеризацию, где каждый объект принадлежит нескольким кластерам с разными степенями принадлежности. 
# Алгоритм настроен на два кластера, максимальное количество итераций — 300, и используется параметр m=2, 
# который управляет жесткостью кластеризации (обычно в диапазоне от 1.5 до 2).

# СМОТРЕТЬ main.py EXAMPLES/FCM_for_data_clustering.png

# ======================================================================================================

fcm = FCM(n_clusters=2, m=2, max_iter=300, random_state=0)

fcm.fit(scaled_data)

cluster_labels = fcm.predict(scaled_data)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].scatter(scaled_data[:, 18], scaled_data[:,19], c=known_labels)
axes[0].scatter(fcm.centers[:, 18], fcm.centers[:, 19],
                s = 100, c = 'black', label = 'Центроиды')
axes[0].set_title(f'Реальные метки кластеров', fontsize=16)

axes[1].scatter(scaled_data[:, 18], scaled_data[:,19], c=cluster_labels, cmap=plt.cm.Set1)
axes[1].scatter(fcm.centers[:, 18], fcm.centers[:, 19],
                s = 100, c = 'black', label = 'Центроиды')
axes[1].set_title(f'Метки кластеров, найденные алгоритмом', fontsize=16)

plt.tight_layout()
plt.show()

# ======================================================================================================
# Делается всё то же самое, но для FCM

# СМОТРЕТЬ EXAMPLES/FCM_visualization_results_clustering.png

# ======================================================================================================

evaluate_clustering(known_labels, cluster_labels)
perform_visualization(scaled_data, cluster_labels, fcm.centers)

# ======================================================================================================
# DBSCAN
# Перебор значений параметров алгоритма DBSCAN по сетке ParameterGrid

# В коде используется алгоритм DBSCAN (Density-Based Spatial Clustering of Applications with Noise) для 
# кластеризации данных с автоматическим подбором параметров, таких как eps (максимальное расстояние между 
# точками в одном кластере) и min_samples (минимальное количество точек для формирования кластера). 
# Цель — выбрать такие параметры, которые обеспечат наилучшее качество кластеризации, измеряемое с помощью 
# silhouette score.

# СМОТРЕТЬ EXAMPLES/DBSCAN_for_data_clustering_with_automatic_selection_of_parameters.png

# ======================================================================================================

param_grid = {
    'eps': [i / 10 for i in range(1, 10)],
    'min_samples': [i for i in range(3, 29)]
}

results = []

for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(scaled_data)

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(scaled_data, labels)
    else:
        silhouette_avg = -1

    results.append({
        'eps': params['eps'],
        'min_samples': params['min_samples'],
        'silhouette_score': silhouette_avg
    })

results_df = pd.DataFrame(results)

best_params = results_df.loc[results_df['silhouette_score'].idxmax()]

plt.figure(figsize=(16, 8))
plt.scatter(scaled_data[:, 18], scaled_data[:, 19], c=labels, cmap='viridis', marker='o', alpha=0.5)
plt.title(f"DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}, silhouette_score={best_params['silhouette_score']:.2f}")

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код выполняет кластеризацию данных с помощью алгоритма DBSCAN, который находит кластеры на основе 
# плотности точек и выявляет шумовые объекты, не относящиеся к никакому кластеру. Он применяет DBSCAN с 
# параметрами eps=0.9 (максимальное расстояние между точками) и min_samples=5 (минимальное количество 
# точек для формирования кластера). Затем создается маска для выделения основных точек, которые принадлежат 
# плотным областям кластеров. Код вычисляет количество кластеров и шумовых объектов, а также оценивает 
# качество кластеризации с использованием различных метрик, таких как однородность, полнота, V-меру, 
# скорректированный индекс Ранда, скорректированную взаимную информацию и коэффициент силуэта. В результате 
# выводится информация о числе кластеров, шумовых объектах и качестве кластеризации.
# ======================================================================================================

db = DBSCAN(eps=0.9, min_samples=5).fit(scaled_data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

labels_true = df['surgical_lesion']

print('Число кластеров: %d' % n_clusters_)
print('Число шумовых объектов: %d' % n_noise_)
print('Homogeneity: %0.3f' % homogeneity_score(labels_true, labels))
print('Completeness: %0.3f' % completeness_score(labels_true, labels))
print('V-measure: %0.3f' % v_measure_score(labels_true, labels))
print('Adjusted Rand Index: %0.3f' % adjusted_rand_score(labels_true, labels))
print('Adjusted Mutual Information: %0.3f'
      % adjusted_mutual_info_score(labels_true, labels))
print('Silhouette score: %0.3f' % silhouette_score(scaled_data, labels))

# ======================================================================================================
# Делается всё то же самое, но для DBSCAN

# СМОТРЕТЬ EXAMPLES/DBSCAN_visualization_results_clustering.png

# ======================================================================================================

evaluate_clustering(known_labels, labels)
perform_visualization(scaled_data, labels)

# ======================================================================================================
# Код выполняет чтение данных из CSV-файла и их предварительную обработку. Выполняется масштабирование 
# данных с использованием MinMaxScaler, который преобразует данные в диапазон от 0 до 1. При этом из 
# DataFrame исключаются столбцы 'Unnamed' и 'y', и оставшиеся данные преобразуются в numpy-массив для 
# дальнейшей обработки. В конце выводятся первые несколько строк DataFrame с помощью метода head(), 
# чтобы проверить результат.
# ======================================================================================================

df = pd.read_csv('WORK_FILES/Epileptic Seizure Recognition.csv')

scaled_data = MinMaxScaler().fit_transform(df.drop(columns=['Unnamed', 'y']).to_numpy())

print(df.head())

# ============================= Иерархическая кластеризация ============================================
# Код выполняет иерархическую кластеризацию данных с использованием метода ward, который минимизирует 
# сумму квадратов отклонений внутри кластера, и метрики расстояния euclidean для вычисления расстояний 
# между объектами. Функция linkage строит матрицу расстояний, которая описывает, как объекты объединяются 
# в кластеры на каждом шаге и какое расстояние между ними. Результатом является матрица, в которой для 
# каждой итерации кластеризации указаны два объединённых кластера, расстояние между ними и количество 
# объектов в образовавшемся кластере.
# ======================================================================================================

distance_matrix = linkage(scaled_data, method='ward', metric='euclidean')

df_good_looking = pd.DataFrame(distance_matrix, columns=['Cluster 1', 'Cluster 2', 'Distance', 'Size'])
print(df_good_looking)

# ======================================================================================================
# Код строит дендрограмму для визуализации иерархической кластеризации. Он создаёт фигуру размером 15x30 
# дюймов с белым фоном, а затем использует функцию dendrogram для построения графика, где метки для объектов 
# берутся из столбца 'y' DataFrame. Дендрограмма отображается с ориентацией слева направо и размером шрифта 
# 12 для меток. После этого вызывается plt.tight_layout() для оптимального размещения элементов, и с помощью 
# plt.show() график выводится на экран.
# ======================================================================================================

fig = plt.figure(figsize=(15, 30))
fig.patch.set_facecolor('white')

R = dendrogram(distance_matrix,
               labels = df['y'].to_numpy(),
               orientation='left',
               leaf_font_size=12)

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код вычисляет коэффициент силуэта для разных количеств кластеров и строит график зависимости этого 
# показателя от числа кластеров. В цикле от 2 до 10 для каждого количества кластеров (от 2 до 10) 
# используется функция fcluster, чтобы разделить данные на кластеры, а затем вычисляется среднее 
# значение коэффициента силуэта с помощью silhouette_score. Этот коэффициент оценивает, насколько 
# хорошо объекты сгруппированы внутри кластеров и насколько они отделены от других кластеров. Значения 
# коэффициента силуэта для каждого числа кластеров сохраняются в список Silh. Далее строится график, 
# где на оси X отображается количество кластеров, а на оси Y — соответствующие значения силуэта. График 
# помогает визуализировать, какое количество кластеров даёт наилучший результат по этому индикатору.
# ======================================================================================================

Silh = []

for i in range(2, 11):
    cluster_labels = fcluster(distance_matrix, i, criterion='maxclust')

    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    Silh.append(silhouette_avg)

fig = plt.figure(figsize = (16, 8))

plt.plot(range(2, 11), Silh)
plt.title('Индекс кластерного анализа (Silhouette score)')
plt.xlabel('Число кластеров')
plt.ylabel('Silhouette score')

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код выполняет кластеризацию данных и визуализацию результатов. Сначала с помощью функции fcluster 
# создаются метки кластеров для данных, разделённых на 2 кластера (второй аргумент — 2), используя 
# критерий maxclust для максимального количества кластеров. Затем метки кластеров передаются в функцию 
# perform_visualization, которая визуализирует результаты кластеризации с использованием четырёх различных 
# методов: t-SNE, UMAP, TriMAP и PaCMAP.

# СМОТРЕТЬ EXAMPLES/Second_visualization_results.png

# ======================================================================================================

cluster_labels = fcluster(distance_matrix, 2, criterion='maxclust')

# perform_visualization(scaled_data, cluster_labels) РАСКОММЕНТИРОВАТЬ ДЛЯ ВИЗУАЛИЗАЦИИ ПОТОМ, СЛИШКОМ ТЯЖЕЛО ДЛЯ НОУТА !!!!!!!!!!

# ================================== Итерационные алгоритмы ============================================
# Код выполняет кластеризацию данных с использованием алгоритма KMeans для различных чисел кластеров 
# (от 2 до 10) и рассчитывает два показателя: коэффициент силуэта и внутрикластерное сцепление (WCSS). 
# Для каждого числа кластеров вычисляются метки кластеров и сохраняются значения WCSS и коэффициента 
# силуэта. Затем строится график, который отображает два подграфика: один с зависимостью коэффициента 
# силуэта от числа кластеров, а второй — с зависимостью WCSS (метод локтя) от числа кластеров. Графики 
# помогают выбрать оптимальное количество кластеров для модели.

# СМОТРЕТЬ 

# ======================================================================================================

WCSS = []
Silh = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)

    cluster_labels = kmeans.fit_predict(scaled_data)

    WCSS.append(kmeans.inertia_)
    Silh.append(silhouette_score(scaled_data, cluster_labels, metric='euclidean'))

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].plot(range(2, 11), Silh)
ax[0].set_title('Индекс кластерного анализа (Silhouette score)')
ax[0].set_xlabel('Число кластеров')
ax[0].set_ylabel('Silhouette score')

ax[1].plot(range(2, 11), WCSS)
ax[1].set_title('Метод локтя (elbow method)')
ax[1].set_xlabel('Число кластеров')
ax[1].set_ylabel('WCSS')

plt.tight_layout()
plt.show()

# ======================================================================================================
# KMeans

# Код выполняет кластеризацию данных с использованием алгоритма KMeans с заданным числом кластеров (3). 
# Алгоритм инициализирует центроиды с помощью метода 'k-means++' для более быстрой сходимости, 
# выполняется до 300 итераций с 10 различными запусками для нахождения наилучшего решения. Затем, с 
# помощью метода fit_predict, данные классифицируются в 3 кластера, и метки кластеров сохраняются в 
# cluster_labels. Наконец, вызывается функция perform_visualization, которая визуализирует результаты 
# кластеризации, включая центроиды кластеров, используя переданные данные и метки кластеров.

# СМОТРЕТЬ EXAMPLES/KMeans_SECOND_visualization_results_clustering.png

# ======================================================================================================

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)

cluster_labels = kmeans.fit_predict(scaled_data)

# perform_visualization(scaled_data, cluster_labels, kmeans.cluster_centers_) РАСКОММЕНТИРОВАТЬ ПОТОМ, МНОГО ТРЕБУЕТ МОЩНОСТЕЙ!!!!!!!!

# ======================================================================================================
# FCM

# Код выполняет кластеризацию данных с использованием нечеткого кластеризатора FCM (Fuzzy C-Means) для 
# разных чисел кластеров (от 2 до 10). Для каждого числа кластеров создается объект FCM с параметрами, 
# такими как степень нечеткости m=1.3, максимальное количество итераций max_iter=300 и фиксированное 
# случайное состояние. Затем применяется метод fit для обучения модели, а метод predict используется для 
# получения меток кластеров. Коэффициент силуэта для каждой кластеризации вычисляется и сохраняется в 
# список Silh. После этого строится график, который показывает зависимость коэффициента силуэта от числа 
# кластеров, помогая оценить, при каком числе кластеров кластеризация дает наилучший результат

# СМОТРЕТЬ EXAMPLES/Silh_SECOND.png

# ======================================================================================================

Silh = []

for i in range(2, 11):
    fcm = FCM(n_clusters=i, m=1.3, max_iter=300, random_state=0)
    
    fcm.fit(scaled_data)

    cluster_labels = fcm.predict(scaled_data)

    Silh.append(silhouette_score(scaled_data, cluster_labels))

fig = plt.figure(figsize = (16, 8))

plt.plot(range(2, 11), Silh)
plt.title('Индекс кластерного анализа (Silhouette score)')
plt.xlabel('Число кластеров')
plt.ylabel('Silhouette score')

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код выполняет кластеризацию данных с использованием нечеткого метода FCM (Fuzzy C-Means) для разделения 
# на 2 кластера. Параметр m=2 задаёт степень нечеткости, что влияет на степень "размытия" границ между 
# кластерами. Алгоритм обучается с помощью метода fit, а затем метки кластеров получаются с помощью 
# метода predict. После этого вызывается функция perform_visualization, которая визуализирует результаты 
# кластеризации, включая центры кластеров, передавая данные, метки кластеров и центры кластеров, 
# полученные из модели.

# СМОТРЕТЬ EXAMPLES/FCM_SECOND_visualization_results_clustering.png

# ======================================================================================================

fcm = FCM(n_clusters=2, m=2, max_iter=300, random_state=0)

fcm.fit(scaled_data)

cluster_labels = fcm.predict(scaled_data)

perform_visualization(scaled_data, cluster_labels, fcm.centers)

# ======================================================================================================
# DBSCAN

# Код выполняет кластеризацию данных с использованием алгоритма DBSCAN и ищет наилучшие параметры для 
# этого метода с помощью перебора значений eps и min_samples. Параметры eps и min_samples задаются в сетке 
# значений, определённой в param_grid. С помощью ParameterGrid происходит перебор всех возможных комбинаций 
# этих параметров. Для каждой комбинации создаётся объект DBSCAN, который обучается на данных с параметрами 
# eps и min_samples. Затем рассчитывается коэффициент силуэта для полученных кластеров. Если количество 
# кластеров больше одного, рассчитывается коэффициент силуэта; если кластеров нет, значение коэффициента 
# силуэта устанавливается равным -1. Результаты сохраняются в список, который затем преобразуется в 
# DataFrame для удобства анализа.

# СМОТРЕТЬ EXAMPLES/DBSCAN_SECOND_for_data_clustering_with_automatic_selection_of_parameters.png

# ======================================================================================================


param_grid = {
    'eps': [i / 10 for i in range(1, 10)],
    'min_samples': [i for i in range(2, 10)]
}

results = []

for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(scaled_data)

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(scaled_data, labels)
    else:
        silhouette_avg = -1

    results.append({
        'eps': params['eps'],
        'min_samples': params['min_samples'],
        'silhouette_score': silhouette_avg
    })

results_df = pd.DataFrame(results)

best_params = results_df.loc[results_df['silhouette_score'].idxmax()]

plt.figure(figsize=(16, 8))
plt.scatter(scaled_data[:, 18], scaled_data[:, 19], c=labels, cmap='viridis', marker='o', alpha=0.5)
plt.title(f"DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}, silhouette_score={best_params['silhouette_score']:.2f}")

plt.tight_layout()
plt.show()

# ======================================================================================================
# Код выполняет кластеризацию данных с использованием алгоритма DBSCAN, а затем оценивает результаты 
# кластеризации с помощью различных метрик (такое уже было до этого).
# ======================================================================================================

db = DBSCAN(eps=0.9, min_samples=4).fit(scaled_data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

labels_true = df['y']

print('Число кластеров: %d' % n_clusters_)
print('Число шумовых объектов: %d' % n_noise_)
print('Homogeneity: %0.3f' % homogeneity_score(labels_true, labels))
print('Completeness: %0.3f' % completeness_score(labels_true, labels))
print('V-measure: %0.3f' % v_measure_score(labels_true, labels))
print('Adjusted Rand Index: %0.3f' % adjusted_rand_score(labels_true, labels))
print('Adjusted Mutual Information: %0.3f'
      % adjusted_mutual_info_score(labels_true, labels))
print('Silhouette score: %0.3f' % silhouette_score(scaled_data, labels))

perform_visualization(scaled_data, labels) # ОТКЛЮЧИТЬ ПОТОМ, МНОГО ТРАТИТ!!!!

# ======================================== MAMMOTH ФАЙЛ ================================================
# Тут не нужно много расписывать. Происходит ровно то же самое, что и до этого, просто читайте код и 
# смотрите логи, а также файлы визуализации.
# ======================================================================================================

df = pd.read_csv('WORK_FILES/mammoth.csv', nrows=100000).astype(np.float32)

scaled_data = MinMaxScaler().fit_transform(df.to_numpy())

df.head()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['x'], df['y'], df['z'], marker='o', s=0.5)

ax.set_title('Трехмерный мамонт')

plt.show()


# ВЫЛЕЗАЕТ ОШИБКА - НУЖНО ИСПРАВИТЬ ПОТОМ

# warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak on Windows.*")

# WCSS = []
# Silh = []

# for i in range(2, 21):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++',
#                     max_iter = 300, n_init = 10, random_state = 0)

#     cluster_labels = kmeans.fit_predict(scaled_data)

#     WCSS.append(kmeans.inertia_)
#     Silh.append(silhouette_score(scaled_data, cluster_labels, metric='euclidean'))

# # =================================== Silh и Elbow ===================================================

# fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# ax[0].plot(range(2, 21), Silh)
# ax[0].set_title('Индекс кластерного анализа (Silhouette score)')
# ax[0].set_xlabel('Число кластеров')
# ax[0].set_ylabel('Silhouette score')

# ax[1].plot(range(2, 21), WCSS)
# ax[1].set_title('Метод локтя (elbow method)')
# ax[1].set_xlabel('Число кластеров')
# ax[1].set_ylabel('WCSS')

# plt.tight_layout()
# plt.show()

# # =========================================== KMeans ================================================
# # ТУТ ДВА КЛАСТЕРА

# kmeans = KMeans(n_clusters = 2, init = 'k-means++',
#                     max_iter = 300, n_init = 10, random_state = 0)

# cluster_labels = kmeans.fit_predict(scaled_data)

# perform_visualization(scaled_data, cluster_labels, kmeans.cluster_centers_)

# # =========================================== KMeans ================================================
# # ТУТ 6 КЛАСТЕРОВ

# kmeans = KMeans(n_clusters = 6, init = 'k-means++',
#                     max_iter = 300, n_init = 10, random_state = 0)

# cluster_labels = kmeans.fit_predict(scaled_data)

# perform_visualization(scaled_data, cluster_labels, kmeans.cluster_centers_)

# # =========================================== KMeans ================================================
# # ТУТ 10 КЛАСТЕРОВ

# kmeans = KMeans(n_clusters = 10, init = 'k-means++',
#                     max_iter = 300, n_init = 10, random_state = 0)

# cluster_labels = kmeans.fit_predict(scaled_data)

# perform_visualization(scaled_data, cluster_labels, kmeans.cluster_centers_)