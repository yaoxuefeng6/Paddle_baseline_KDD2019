import paddle.fluid as fluid
import math

user_profile_dim = 65
slot_1 = [0, 1, 2, 3, 4, 5]
slot_2 = [6, 7, 8, 9]
slot_3 = [10, 11, 12, 13, 14]

def ctr_deepfm_dataset(user_profile, context_feature, label,
                       embedding_size, sparse_feature_dim):
    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            # you need to patch https://github.com/PaddlePaddle/Paddle/pull/14190
            # if you want to set is_distributed to True
            is_distributed=False,
            size=[sparse_feature_dim, embedding_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                       initializer=fluid.initializer.Uniform()))

    sparse_embed_seq = list(map(embedding_layer, context_feature))

    w = fluid.layers.create_parameter(
        shape=[65, 65], dtype='float32',
        name="w_fm")

    user_emb_list = []
    user_profile_emb = fluid.layers.matmul(user_profile, w)
    user_emb_list.append(user_profile_emb)

    w1 = fluid.layers.create_parameter(shape=[65, 16], dtype='float32', name="w_1")
    w2 = fluid.layers.create_parameter(shape=[65, 16], dtype='float32', name="w_2")
    w3 = fluid.layers.create_parameter(shape=[65, 16], dtype='float32', name="w_3")
    user_profile_emb_1 = fluid.layers.matmul(user_profile, w1)
    user_profile_emb_2 = fluid.layers.matmul(user_profile, w2)
    user_profile_emb_3 = fluid.layers.matmul(user_profile, w3)

    sparse_embed_seq_1 = embedding_layer(context_feature[slot_1[0]])
    sparse_embed_seq_2 = embedding_layer(context_feature[slot_2[0]])
    sparse_embed_seq_3 = embedding_layer(context_feature[slot_3[0]])
    for i in slot_1[1:-1]:
        sparse_embed_seq_1 = fluid.layers.elementwise_add(sparse_embed_seq_1, embedding_layer(context_feature[slot_1[i]]))
    for i in slot_2[1:-1]:
        sparse_embed_seq_2 = fluid.layers.elementwise_add(sparse_embed_seq_2, embedding_layer(context_feature[i]))
    for i in slot_3[1:-1]:
        sparse_embed_seq_3 = fluid.layers.elementwise_add(sparse_embed_seq_3, embedding_layer(context_feature[i]))

    ele_product_1 = fluid.layers.elementwise_mul(user_profile_emb_1, sparse_embed_seq_1)
    user_emb_list.append(ele_product_1)
    ele_product_2 = fluid.layers.elementwise_mul(user_profile_emb_2, sparse_embed_seq_2)
    user_emb_list.append(ele_product_2)
    ele_product_3 = fluid.layers.elementwise_mul(user_profile_emb_3, sparse_embed_seq_3)
    user_emb_list.append(ele_product_3)

    ffm_1 = fluid.layers.reduce_sum(ele_product_1, dim=1, keep_dim=True)
    ffm_2 = fluid.layers.reduce_sum(ele_product_2, dim=1, keep_dim=True)
    ffm_3 = fluid.layers.reduce_sum(ele_product_3, dim=1, keep_dim=True)



    concated_ori = fluid.layers.concat(sparse_embed_seq + user_emb_list, axis=1)
    concated = fluid.layers.batch_norm(input=concated_ori, name="bn", epsilon=1e-4)

    deep = deep_net(concated)
    linear_term, second_term = fm(concated, 353, 8) #depend on the number of context feature

    predict = fluid.layers.fc(input=[deep, linear_term, second_term, ffm_1, ffm_2, ffm_3], size=2, act="softmax",
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(deep.shape[1])), learning_rate=0.01))

    #similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(predict, min=-15.0, max=15.0), name="similarity_norm")


    cost = fluid.layers.cross_entropy(input=predict, label=label)

    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=label, num_thresholds=2 ** 12, slide_steps=20)
    return avg_cost, auc_var, batch_auc_var, accuracy, predict


def deep_net(concated, lr_x=0.0001):
    fc_layers_input = [concated]
    fc_layers_size = [256, 128, 64, 32, 16]
    fc_layers_act = ["relu"] * (len(fc_layers_size))

    for i in range(len(fc_layers_size)):
        fc = fluid.layers.fc(
            input=fc_layers_input[-1],
            size=fc_layers_size[i],
            act=fc_layers_act[i],
            param_attr=fluid.ParamAttr(learning_rate=lr_x * 0.5))

        fc_layers_input.append(fc)

    return fc_layers_input[-1]


def fm(concated, emb_dict_size, factor_size, lr_x=0.0001):
    linear_term = fluid.layers.fc(input=concated, size=8, act=None, param_attr=fluid.ParamAttr(learning_rate=lr_x))

    emb_table = fluid.layers.create_parameter(shape=[emb_dict_size, factor_size],
                                                  dtype='float32')

    input_mul_factor = fluid.layers.matmul(concated, emb_table)
    input_mul_factor_square = fluid.layers.square(input_mul_factor)
    input_square = fluid.layers.square(concated)
    factor_square = fluid.layers.square(emb_table)
    input_square_mul_factor_square = fluid.layers.matmul(input_square, factor_square)

    second_term = 0.5 * (input_mul_factor_square - input_square_mul_factor_square)

    return linear_term, second_term