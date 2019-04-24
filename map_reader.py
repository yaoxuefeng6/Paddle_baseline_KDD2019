import sys
import json
import paddle.fluid.incubate.data_generator as dg


class MapDataset(dg.MultiSlotDataGenerator):
    def setup(self, sparse_feature_dim):
        self.profile_length = 65
        self.query_feature_list = ["weekday", "hour", "o1", "o2", "d1", "d2"]
        self.plan_feature_list = ["distance", "price", "eta", "transport_mode"]
        self.rank_feature_list = ["plan_rank", "whole_rank", "price_rank", "eta_rank", "distance_rank"]
        self.hash_dim = 1000001
        self.train_idx_ = 2000000
        self.categorical_range_ = range(0, 15)

    def _process_line(self, line):
        instance = json.loads(line)
        profile = instance["profile"]
        user_profile_feature = [0] * self.profile_length
        if len(profile) > 1 or (len(profile) == 1 and profile[0] != 0):
            for p in profile:
                if p >= 1 and p <= 65:
                    user_profile_feature[p - 1] = 1
        #for test
        #user_profile_feature[64] = 1
        context_feature = []
        """
        try:
            if len(profile) > 1 or (len(profile) == 1 and profile[0] != 0):
                for p in profile:
                    if p >= 1 and p <= 65:
                        user_profile_feature[p-1] = 1
        except IndexError as e:
            pass
            #print("INDEX ERROR: ", e)
        finally:
            pass
            print(profile)
        """
        query = instance["query"]
        plan = instance["plan"]
        for fea in self.query_feature_list:
            context_feature.append([hash(fea + str(query[fea])) % self.hash_dim])
        for fea in self.plan_feature_list:
            context_feature.append([hash(fea + str(plan[fea])) % self.hash_dim])
        for fea in self.rank_feature_list:
            context_feature.append([hash(fea + str(instance[fea])) % self.hash_dim])

        label = [int(instance["label"])]
        #print(user_profile_feature)
        #print(context_feature)
        #print(label)
        return user_profile_feature, context_feature, label

    def infer_reader(self, filelist, batch, buf_size):
        print(filelist)

        def local_iter():
            for fname in filelist:
                with open(fname.strip(), "r") as fin:
                    for line in fin:
                        dense_feature, sparse_feature, label = self._process_line(line)
                        # yield dense_feature, sparse_feature, label
                        yield [dense_feature] + sparse_feature + [label]

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.shuffle(
                local_iter, buf_size=buf_size),
            batch_size=batch)
        return batch_iter

    def test_reader(self, filelist, batch, buf_size):
        print(filelist)

        def local_iter():
            for fname in filelist:
                with open(fname.strip(), "r") as fin:
                    for line in fin:
                        dense_feature, sparse_feature, label = self._process_line(line)
                        # yield dense_feature, sparse_feature, label
                        yield [dense_feature] + sparse_feature + [label]

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.buffered(
                local_iter, size=buf_size),
            batch_size=batch)
        return batch_iter

    def generate_sample(self, line):
        def data_iter():
            dense_feature, sparse_feature, label = self._process_line(line)
            feature_name = ["user_profile"]
            for idx in self.categorical_range_:
                feature_name.append("context" + str(idx))
            feature_name.append("label")
            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return data_iter


if __name__ == "__main__":
    map_dataset = MapDataset()
    map_dataset.setup(int(sys.argv[1]))
    map_dataset.run_from_stdin()
