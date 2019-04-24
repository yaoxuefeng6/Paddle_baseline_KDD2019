import argparse
import logging
import numpy as np
# disable gpu training for this example
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--model_path',
        type=str,
        #required=True,
        default='models',
        help="The path of model parameters gz file")
    parser.add_argument(
        '--data_path',
        type=str,
        required=False,
        help="The path of the dataset to infer")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=16,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help="The size for embedding layer (default:1000001)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")

    return parser.parse_args()

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def data2tensor(data, place):
    feed_dict = {}
    dense = data[0]
    sparse = data[1:-1]
    y = data[-1]
    dense_data = np.array([x[0] for x in data]).astype("float32")
    dense_data = dense_data.reshape([-1, 65])
    feed_dict["user_profile"] = dense_data
    for i in range(15):##################### temporaly note this
        sparse_data = to_lodtensor([x[1 + i] for x in data], place)
        feed_dict["context" + str(i)] = sparse_data

    y_data = np.array([x[-1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    feed_dict["label"] = y_data
    return feed_dict

def test():
    args = parse_args()

    place = fluid.CPUPlace()
    test_scope = fluid.core.Scope()

    # filelist = ["%s/%s" % (args.data_path, x) for x in os.listdir(args.data_path)]
    from map_reader import MapDataset
    map_dataset = MapDataset()
    map_dataset.setup(args.sparse_feature_dim)
    exe = fluid.Executor(place)

    # whole_filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
    whole_filelist = ["./out/normed_test_session.txt"]
    test_files = whole_filelist[int(0.0 * len(whole_filelist)):int(1.0 * len(whole_filelist))]

    def set_zero(var_name):
        param = inference_scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype("int64")
        param.set(param_array, place)

    epochs = 1

    for i in range(epochs):
        cur_model_path = args.model_path + "/epoch" + str(10) + ".model"
        with open("./testres/res" + str(i + 8), 'w') as r:
            with fluid.scope_guard(test_scope):
                [inference_program, feed_target_names, fetch_targets] = \
                    fluid.io.load_inference_model(cur_model_path, exe)

                test_reader = map_dataset.test_reader(test_files, 1000, 100000)
                k = 0
                for batch_id, data in enumerate(test_reader()):
                    feed_dict = data2tensor(data, place)
                    loss_val, auc_val, accuracy, predict, _ = exe.run(inference_program,
                                                feed=feed_dict,
                                                fetch_list=fetch_targets, return_numpy=False)
                    #if k % 100 == 0:
                        #print(np.array(predict))
                        #x = np.array(predict)
                        #print(x.shape)
                    #k += 1
                    x = np.array(predict)
                    for j in range(x.shape[0]):
                        r.write(str(x[j][1]))
                        r.write("\n")


if __name__ == '__main__':
    test()
