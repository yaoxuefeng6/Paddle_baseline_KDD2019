import os, sys, time, random, csv, datetime, json
import pandas as pd
import numpy as np
import argparse
import logging
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)

TRAIN_QUERIES_PATH = "./data_set_phase1/train_queries.csv"
TRAIN_PLANS_PATH = "./data_set_phase1/train_plans.csv"
TRAIN_CLICK_PATH = "./data_set_phase1/train_clicks.csv"
PROFILES_PATH = "./data_set_phase1/profiles.csv"

O1_MIN = 115.47
O1_MAX = 117.29

O2_MIN = 39.46
O2_MAX = 40.97

D1_MIN = 115.44
D1_MAX = 117.37

D2_MIN = 39.46
D2_MAX = 40.96

DISTANCE_MIN = 1.0
DISTANCE_MAX = 225864.0
THRESHOLD_DIS = 40000.0

PRICE_MIN = 200.0
PRICE_MAX = 92300.0
THRESHOLD_PRICE = 20000

ETA_MIN = 1.0
ETA_MAX = 72992.0
THRESHOLD_ETA = 10800.0


def parse_args():
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default='./data_set_phase/',
        help="The path of training information")
    parser.add_argument(
        '--preprocessed_data_path',
        type=str,
        default='./out_data/',
        help="The path of preprocessed out data")
    return parser.parse_args()


def build_norm_feature():
    with open("./out/normed_train.txt", 'w') as nf:
        with open("./out/train.txt", 'r') as f:
            for line in f:
                cur_map = json.loads(line)

                if cur_map["plan"]["distance"] > THRESHOLD_DIS:
                    cur_map["plan"]["distance"] = int(THRESHOLD_DIS)
                elif cur_map["plan"]["distance"] > 0:
                    cur_map["plan"]["distance"] = int(cur_map["plan"]["distance"] / 500)

                if cur_map["plan"]["price"] and cur_map["plan"]["price"] > THRESHOLD_PRICE:
                    cur_map["plan"]["price"] = int(THRESHOLD_PRICE)
                elif not cur_map["plan"]["price"] or cur_map["plan"]["price"] < 0:
                    cur_map["plan"]["price"] = 0
                else:
                    cur_map["plan"]["price"] = int(cur_map["plan"]["price"] / 100)

                if cur_map["plan"]["eta"] > THRESHOLD_ETA:
                    cur_map["plan"]["eta"] = int(THRESHOLD_ETA)
                elif cur_map["plan"]["eta"] > 0:
                    cur_map["plan"]["eta"] = int(cur_map["plan"]["eta"] / 120)

                # o1
                if cur_map["query"]["o1"] > O1_MAX:
                    cur_map["query"]["o1"] = int((O1_MAX - O1_MIN) / 0.02 + 1)
                elif cur_map["query"]["o1"] < O1_MIN:
                    cur_map["query"]["o1"] = 0
                else:
                    cur_map["query"]["o1"] = int((cur_map["query"]["o1"] - O1_MIN) / 0.02)

                # o2
                if cur_map["query"]["o2"] > O2_MAX:
                    cur_map["query"]["o2"] = int((O2_MAX - O2_MIN) / 0.02 + 1)
                elif cur_map["query"]["o2"] < O2_MIN:
                    cur_map["query"]["o2"] = 0
                else:
                    cur_map["query"]["o2"] = int((cur_map["query"]["o2"] - O2_MIN) / 0.02)

                # d1
                if cur_map["query"]["d1"] > D1_MAX:
                    cur_map["query"]["d1"] = int((D1_MAX - D1_MIN) / 0.02 + 1)
                elif cur_map["query"]["d1"] < D1_MIN:
                    cur_map["query"]["d1"] = 0
                else:
                    cur_map["query"]["d1"] = int((cur_map["query"]["d1"] - D1_MIN) / 0.02)

                # d2
                if cur_map["query"]["d2"] > D2_MAX:
                    cur_map["query"]["d2"] = int((D2_MAX - D2_MIN) / 0.02 + 1)
                elif cur_map["query"]["d2"] < D2_MIN:
                    cur_map["query"]["d2"] = 0
                else:
                    cur_map["query"]["d2"] = int((cur_map["query"]["d2"] - D2_MIN) / 0.02)

                cur_json_instance = json.dumps(cur_map)
                nf.write(cur_json_instance + '\n')


def preprocess():
    """
    Construct the train data indexed by session id and mode id jointly. Convert all the raw features (user profile,
    od pair, req time, click time, eta, price, distance, transport mode) to one-hot ids used for
    embedding. We split the one-hot features into two categories: user feature and context feature for
    better understanding of FFM algorithm.
    Note that the user profile is already provided by one-hot encoded form, we convert it back to the
    ids for unity with the context feature and easily using of PaddlePaddle embedding layer. Given the
    train clicks data, we label each train instance with 1 or 0 depend on if this instance is clicked or
    not.
    :return:
    """
    #args = parse_args()

    train_data_dict = {}
    with open(TRAIN_QUERIES_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        train_index_list = []
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            if line[0] == "": continue
            if line[1] == "":
                train_index_list.append(line[0] + "_0")
            else:
                train_index_list.append(line[0] + "_" + line[1])

            train_index = line[0]
            train_data_dict[train_index] = {}
            train_data_dict[train_index]["pid"] = line[1]
            train_data_dict[train_index]["query"] = {}

            reqweekday = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%w")
            reqhour = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%H")

            train_data_dict[train_index]["query"].update({"weekday":reqweekday})
            train_data_dict[train_index]["query"].update({"hour":reqhour})

            o = line[3].split(',')
            o_first = o[0]
            o_second = o[1]
            train_data_dict[train_index]["query"].update({"o1":float(o_first)})
            train_data_dict[train_index]["query"].update({"o2":float(o_second)})

            d = line[4].split(',')
            d_first = d[0]
            d_second = d[1]
            train_data_dict[train_index]["query"].update({"d1":float(d_first)})
            train_data_dict[train_index]["query"].update({"d2":float(d_second)})

    plan_map = {}
    plan_data = pd.read_csv(TRAIN_PLANS_PATH)
    for index, row in plan_data.iterrows():
        plans_str = row['plans']
        plans_list = json.loads(plans_str)
        session_id = str(row['sid'])
        # train_data_dict[session_id]["plans"] = []
        plan_map[session_id] = plans_list

    profile_map = {}
    with open(PROFILES_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            profile_map[line[0]] = [i for i in range(len(line)) if line[i] == "1.0"]

    session_click_map = {}
    with open(TRAIN_CLICK_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            if line[0] == "" or line[1] == "" or line[2] == "":
                continue
            session_click_map[line[0]] = line[2]
    #return train_data_dict, profile_map, session_click_map, plan_map
    generate_sparse_features(train_data_dict, profile_map, session_click_map, plan_map)


def generate_sparse_features(train_data_dict, profile_map, session_click_map, plan_map):
    if not os.path.isdir("./out/"):
        os.mkdir("./out/")
    with open(os.path.join("./out/", "train.txt"), 'w') as f_train:
        for session_id, plan_list in plan_map.items():
            if session_id not in train_data_dict:
                continue
            cur_map = train_data_dict[session_id]
            if cur_map["pid"] != "":
                cur_map["profile"] = profile_map[cur_map["pid"]]
            else:
                cur_map["profile"] = [0]
            del cur_map["pid"]
            flag_click = False
            for plan in plan_list:
                if ("transport_mode" in plan) and (session_id in session_click_map) and (
                        int(plan["transport_mode"]) == int(session_click_map[session_id])):
                    cur_map["plan"] = plan
                    cur_map["label"] = 1
                    flag_click = True
                    # print("label is 1")
                else:
                    cur_map["plan"] = plan
                    cur_map["label"] = 0

                cur_json_instance = json.dumps(cur_map)
                f_train.write(cur_json_instance + '\n')
            if not flag_click:
                cur_map["plan"]["distance"] = -1
                cur_map["plan"]["price"] = -1
                cur_map["plan"]["eta"] = -1
                cur_map["plan"]["transport_mode"] = 0
                cur_map["label"] = 1
                cur_json_instance = json.dumps(cur_map)
                f_train.write(cur_json_instance + '\n')
            else:
                cur_map["plan"]["distance"] = -1
                cur_map["plan"]["price"] = -1
                cur_map["plan"]["eta"] = -1
                cur_map["plan"]["transport_mode"] = 0
                cur_map["label"] = 0
                cur_json_instance = json.dumps(cur_map)
                f_train.write(cur_json_instance + '\n')


    build_norm_feature()


if __name__ == "__main__":
    preprocess()
