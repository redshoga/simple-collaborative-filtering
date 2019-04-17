import numpy as np

# Cosine similarity matrix
# https://qiita.com/fam_taro/items/dac3b1bcfc01461a0120
def cos_sim_matrix(matrix):
    d = matrix @ matrix.T
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    return d / norm / norm.T

if __name__ == "__main__":
    # sample data: http://kysmo.hatenablog.jp/entry/2018/10/31/170330
    behavior_matrix = np.array([
        [0,0,0,1,3,5,0,0,0,1,0],
        [0,1,1,3,0,3,3,1,0,0,0],
        [0,1,5,4,1,1,1,1,0,0,0],
        [0,0,0,5,3,1,1,0,1,5,0],
        [1,3,3,3,3,3,0,1,0,0,1],
        [1,1,5,1,3,5,2,1,2,1,0],
        [0,1,2,3,3,3,0,0,0,0,0],
        [0,1,1,1,1,1,0,0,0,0,0]
    ])

    for user_idx, history in enumerate(behavior_matrix):
        sim_vec = cos_sim_matrix(behavior_matrix)[user_idx]
        recommend_matrix = np.dot(np.diag(sim_vec), behavior_matrix)
        recommend = np.sum(recommend_matrix, axis=0)

        print("user:", user_idx)
        print("history:", history)
        print("recommend:", recommend)
