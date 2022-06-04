from collections import Counter

from util import flatten_lists


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)
        # print("真实：",self.golden_tags)
        # print("真实类型：", type(self.golden_tags))
        # print("预测：", self.predict_tags)
        # print("预测类型：", type(self.predict_tags))
        # print(predict_tags)
        # 如果remove_O为true，就去除所有的O标签
        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量

        # 所有的实体种类
        self.tagset = self.entity_number()
        print("实体种类：", self.tagset)

        # 模型预测正确的每类实体数量
        self.correct_entity = self.correct_entity_number()
        print("预测正确的实体数量：", self.correct_entity)
        # 模型预测的每类实体数量
        self.predict_entity = self.predict_entity_number()
        print("模型预测的实体总数量：", self.predict_entity)
        # 数据集中的每类实体总数量
        self.golden_entity = self.golden_entity_number()
        print("测试数据集中实体总数量：", self.golden_entity)
        # 计算精确率
        # 模型预测正确的实体数量/模型预测的实体总数量
        self.precision_scores = self.cal_precision()

        # 计算召回率
        # 模型预测正确的实体数量/数据集中实体总数量
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def entity_number(self):
        tagset = set()
        for i in range(len(self.golden_tags)):
            if self.golden_tags[i] == '[CLS]' or self.golden_tags[i] == '[SEP]' or self.golden_tags[i] == 'O':
                continue
            else:
                tagset.add(self.golden_tags[i][2:])
        return tagset

    def get_golden_index(self):
        # 记录原始标签中实体的index
        golden_tags = self.golden_tags
        begin = None
        end = None
        lock = False
        # 字典表示每种实体的index{LOC : [[6, 9], [10, 12]]}
        golden_dic = {m_type: set() for m_type in list(self.tagset)}
        for index, elem in enumerate(golden_tags):
            if lock is False and elem.startswith('B-'):
                begin = index
                lock = True
                continue
            if lock is True and elem.startswith('E-'):
                end = index
                lock = False
                golden_dic[elem.replace("E-", "")].add((begin, end))
                continue
        return golden_dic

    def correct_entity_number(self):
        """计算每种实体预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dic = {m_type: 0 for m_type in self.tagset}
        golden_tags = self.golden_tags
        predict_tags = self.predict_tags
        golden_indexes = self.get_golden_index()
        for tag in self.tagset:
            indexes = golden_indexes[tag]
            for index in indexes:
                if golden_tags[index[0]: index[1] + 1] == predict_tags[index[0]: index[1] + 1]:
                    correct_dic[tag] += 1
        return correct_dic

    # 模型预测的每类实体的数量
    # 必须以B开头，I为中间，E结尾，这里漏掉了以I为中间，以E结尾
    def predict_entity_number(self):
        predict_dic = {}
        predict_tags = self.predict_tags
        for i in range(len(predict_tags)):
            # 如果以B开头
            if predict_tags[i].startswith("B"):
                tag = predict_tags[i].replace("B-", "")
                for j in range(i + 1, len(predict_tags)):
                    # 如果以E结尾,对应tag也一致,那么对应预测的实体+1
                    if predict_tags[j].startswith('E') and predict_tags[j].replace("E-", "") == tag:
                        if tag not in predict_dic:
                            predict_dic[tag] = 1
                        else:
                            predict_dic[tag] += 1
                    # 如果以I-开头，对应标签一致，continue
                    elif predict_tags[j].startswith('I') and predict_tags[j].replace("I-", "") == tag:
                        continue
                    #     其他情况，跳出循环
                    else:
                        break
            else:
                continue

        # print(predict_dic)
        return predict_dic

    # 数据集中的每类实体总数量
    # 必须以B开头，E结尾，这里以B开头肯定以E结尾
    def golden_entity_number(self):
        golden_dic = {}
        golden_tags = self.golden_tags
        for i in range(len(golden_tags)):
            if golden_tags[i].startswith("B"):
                tag = golden_tags[i].replace("B-", "")
                # print(tag)
                if tag not in golden_dic:
                    golden_dic[tag] = 1
                else:
                    golden_dic[tag] += 1
        # print(golden_dic)
        return golden_dic

    def cal_precision(self):
        # 模型预测正确的实体数量/模型预测的实体总数量
        precision_scores = {}
        for tag in self.tagset:
            if self.predict_entity.get(tag, 0) == 0 or tag not in self.correct_entity:
                continue
            print(self.correct_entity.get(tag, 0))
            print(self.predict_entity.get(tag))
            precision_scores[tag] = self.correct_entity.get(tag, 0) / self.predict_entity.get(tag)

        return precision_scores

    def cal_recall(self):
        # 模型预测正确的实体数量/数据集中实体总数量
        recall_scores = {}
        for tag in self.tagset:
            if self.golden_entity.get(tag, 0) == 0 or tag not in self.correct_entity:
                continue
            print(self.golden_entity[tag])
            recall_scores[tag] = self.correct_entity[tag] / self.golden_entity[tag]

        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores.get(tag, 0), self.recall_scores.get(tag, 0)
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores.get(tag, 0),
                self.recall_scores.get(tag, 0),
                self.f1_scores.get(tag, 0),
                self.golden_entity.get(tag, 0)
            ))

        # 计算并打印平均值
        avg_metrics, total = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            total
        ))
        return avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score']

    def _cal_weighted_average(self):

        weighted_average = {}
        total = 0
        # 所有实体的数量
        for i in self.golden_entity:
            total += self.golden_entity.get(i)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_entity[tag]
            # 准确率 * 数量/总数量
            weighted_average['precision'] += self.precision_scores.get(tag, 0) * size
            weighted_average['recall'] += self.recall_scores.get(tag, 0) * size
            weighted_average['f1_score'] += self.f1_scores.get(tag, 0) * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average, total

    def _remove_Otags(self):

        length = len(self.golden_tags)
        # 去除原始标记中为O的那部分
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O' or self.golden_tags[i] == '[CLS]' or self.golden_tags == '[SEP]']
        # golden_tags为去除了O标签的所有tags
        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]
        # predict_tags也是去除了O标签的所有tags
        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size + 1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
