from __future__ import print_function

from args import parse_args
import os
import paddle.fluid as fluid
import sys
from network_conf import ctr_deepfm_dataset


NUM_CONTEXT_FEATURE = 15
DIM_USER_PROFILE = 65

def train():
    args = parse_args()
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    user_profile = fluid.layers.data(
        name="user_profile", shape=[DIM_USER_PROFILE], dtype='float32')
    context_feature = [
        fluid.layers.data(name="context" + str(i), shape=[1], lod_level=1, dtype="int64")
        for i in range(0, NUM_CONTEXT_FEATURE)]
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    print("ready to network")
    loss, auc_var, batch_auc_var, accuracy, predict = ctr_deepfm_dataset(user_profile, context_feature, label,
                                                        args.embedding_size, args.sparse_feature_dim)

    print("ready to optimizer")
    optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
    optimizer.minimize(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([user_profile] + context_feature + [label])
    pipe_command = "/home/yaoxuefeng/whls/paddle_release_home/python/bin/python  map_reader.py %d" % args.sparse_feature_dim
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(100)
    thread_num = 1
    dataset.set_thread(thread_num)
    whole_filelist = ["./out/normed_train01", "./out/normed_train02", "./out/normed_train03",
                      "./out/normed_train04", "./out/normed_train05", "./out/normed_train06", "./out/normed_train07",
                      "./out/normed_train08",
                      "./out/normed_train09", "./out/normed_train10", "./out/normed_train11"]
    print("ready to epochs")
    epochs = 6
    for i in range(epochs):
        print("start %dth epoch" % i)
        dataset.set_filelist(whole_filelist[:int(len(whole_filelist))])
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[auc_var, accuracy, predict, label],
                               fetch_info=["auc", "accuracy", "predict", "label"],
                               debug=False)
        model_dir = args.model_output_dir + '/epoch' + str(i + 1) + ".model"
        sys.stderr.write("epoch%d finished" % (i + 1))
        fluid.io.save_inference_model(model_dir, [user_profile.name] + [x.name for x in context_feature] + [label.name],
                                      [loss, auc_var, accuracy, predict, label], exe)


if __name__ == '__main__':
    train()
