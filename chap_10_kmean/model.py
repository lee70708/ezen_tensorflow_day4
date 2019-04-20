"""
K-평균
군집화 (Clustring) 문제를 풀기위한 자율 학습 알고리즘
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])

df = pd.DataFrame({
    "x" : [v[0] for v in vectors_set],
    "y" : [v[1] for v in vectors_set],
})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# plt.show()
vectors = tf.constant(vectors_set) # 모든 데이터를 텐서로 옮긴다.
k = 4 # 입력데이터에서 무작위로 k개의 데이터를 선택함. 여기서는 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

# k 개의 데이터 포인트느 2D 텐서로 저장됨
print('------------- 텐서의 구조 보기 ---------------')
print(vectors.get_shape())
"""
(20000, 2)
D0차원의 크기가 2000 개이고, D1의 차원의 크기가 2 (각 점의 x,y 좌표)
"""
print('------------- 센트로이드의 구조 보기 ---------------')
print(centroids.get_shape())
"""
(4, 2)
센트로이드 D0차원의 크기가 4개이고, D1의 차원의 크기가 2인 행렬
"""

expanded_vectors = tf.expand_dims(vectors, 0) # 두 텐서에 차원을 추가
expanded_centroids = tf.expand_dims(centroids, 1)
# 두 텐서를 2차원에서 3차원으로
# 만들어 뺄셈 가능하돌고 크기를 맞추는 작업

assignments = tf.argmin(tf.reduce_mean(tf.square(tf.subtract(expanded_vectors, expanded_centroids)),2),0)
# 각 점에 대한 유클리드 제곱거리 알고리즘의 무한 반복
# 새로운 중신 계산하기
# 매 반복마다 새롭게 그룹화하면서 각 그룹에 해당하는 새로운 중심을 다시 계산

means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),[1, -1])),
                                  reduction_indices=[1]) for c in range(k)], 0)
update_centroids = tf.assign(centroids, means)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for i in range(100):
    _, centroids_values, assignments_values = sess.run([
        update_centroids,
        centroids,
        assignments
    ])
    # assignment_values 텐서의 결과를 확인하기 위한 결과 그림 그리기

data = {"x" : [], "y" : [], "cluster" : []}
for i in range(len(assignments_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignments_values[i])
df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, height=6, hue='cluster', legend=False)
plt.show()

