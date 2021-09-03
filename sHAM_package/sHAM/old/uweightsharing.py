import numpy as np


from sHAM import weightsharing, compressed_nn

class uWeightsharing_NN(weightsharing.Weightsharing_NN):
    def __init__(self, model, clusters_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_uws(self, list_trainable=None, untrainable_per_layers=None, mbkmeans=True):

        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense
        vect_weights = [np.hstack(list_weights[i]).reshape(-1,1) for i in range (d, len(list_weights), self.div)]
        all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
        self.centers = weightsharing.build_clusters(weights=all_vect_weights, cluster=self.clusters, mbkmeans=mbkmeans)
        self.idx_layers = [weightsharing.redefine_weights(list_weights[i], self.centers) for i in range (d, len(list_weights), self.div)]

        if not list_trainable:
            self.untrainable_per_layers = 0
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            self.untrainable_per_layers = untrainable_per_layers
            self.model.set_weights(self.recompose_weight(list_weights, True, untrainable_per_layers))

    def recompose_weight(self, list_weights, trainable_vars=False, untrainable_per_layers=None):
        if not trainable_vars:
            d = self.index_first_dense
            return list_weights[:d]+[(weightsharing.idx_matrix_to_matrix(self.idx_layers[(i-d)//self.div], self.centers)) if i%self.div==0 else (list_weights[i]) for i in range(d,len(list_weights))]
        else:
            div = self.div + untrainable_per_layers
            list_weights = self.trainable_to_weights(self.model.get_weights(), list_weights, untrainable_per_layers)
            d = weightsharing.find_index_first_dense(list_weights)
            return list_weights[:d]+[(weightsharing.idx_matrix_to_matrix(self.idx_layers[(i-d)//div], self.centers)) if i%div==0 else (list_weights[i]) for i in range(d,len(list_weights))]

    def update_centers_and_recompose(self, list_weights_before, lr):
        list_weights = self.model.get_weights()
        div = self.div + self.untrainable_per_layers
        d = weightsharing.find_index_first_dense(list_weights)
        centers_upd = [(weightsharing.centroid_gradient_matrix(self.idx_layers[(i-d)//div], list_weights[i]-list_weights_before[i], self.clusters)) for i in range(d,len(list_weights), div)]
        for c_u in centers_upd:
            self.centers = self.centers + lr * c_u

        if  len(list_weights) == len(self.model.trainable_weights):
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            trainable=[]
            for w in (self.model.trainable_weights):
                trainable.append(w.numpy())
            self.model.set_weights(self.recompose_weight(trainable, True, self.untrainable_per_layers))
