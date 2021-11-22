import os
import numpy as np
from sklearn.svm import LinearSVC
from torchvision import transforms, datasets, utils
import json
import cv2
from skimage.feature import hog
import joblib
import time
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn import svm

train_feature_path = "train/"
test_feature_path = "test/"
model_path = "model/"

data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
image_path = data_root + "/dataset/flower_data"


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# 将字典写入json，方便查阅
def write_json(img_name_label):
    if os.path.exists("class_indices.json"):
        pass
    else:
        json_str = json.dumps(img_name_label, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)


# 利用 trochvision的ImageFolder加载数据
# datasets.ImageFolder返回的是一个列表数据，第一个数据为图像的整个路径，第二个数据是这个图像的标签
# 因为datasets.ImageFolder会自动将不同文件夹下的图片打一个“0,1,2,3....”的标签
def load_set(path):
    dataset = datasets.ImageFolder(root=image_path + "/" + path)
    img_name_label = {}
    for i in range(len(dataset.imgs)):
        img_name = dataset.imgs[i][0].split("/")[-1]
        img_label = dataset.imgs[i][1]
        img_name_label[img_name] = img_label
    # write_json(img_name_label)
    print("图片的字典数据已经加载完成......")
    return img_name_label


def get_pic_picname(path):
    # 访问所有文件夹的所有图片并保存
    img_list = []  # 保存图片本身
    img_name_list = []  # 保存图片的名字
    root = image_path + "/" + path
    for dir in os.listdir(root):
        for file in os.listdir(os.path.join(root, dir)):
            img_name_list.append(file)
            img_root = os.path.join(root + "/" + dir, file)
            img_temp = cv2.imread(img_root, 1)
            img_temp = cv2.resize(img_temp, (128, 128))
            img_list.append(img_temp.copy())
    print("已经获取了图片本身和图片的名字")
    return img_list, img_name_list


def get_feature(img_list, img_name_list, img_name_label, savepath):
    # 提取特征并且保存模型
    for i in range(len(img_list)):
        img = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        hogvector, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                   block_norm='L2-Hys', visualize=True)
        label = img_name_label[img_name_list[i]] # 通过上面字典的方式获取图片标签
        fd = np.concatenate((hogvector, [label])) # 将图片的hog特征和标签串联起来
        fd_name = img_name_list[i] + '.feat'
        makedir(savepath)
        fd_path = os.path.join(savepath, fd_name)
        joblib.dump(fd, fd_path)
    print("数据集的特征和标签已经获取完成。")


def train_testmodel():
    print("开始训练模型了.....")
    # 训练模型
    start_time = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0
    # glob以list的形式将满足要求的文件返回,得到了所有图像的特征和对应的标签
    for feat_path in glob.glob(os.path.join(train_feature_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])

    print("Training a SVM Classifier.")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # clf = LinearSVC(random_state=42)
    # clf = svm.SVC(C=5, gamma=0.05)
    clf = svm.SVC(gamma="scale")  # 没有标准化的准确率是0.431319；   标准化后准确率是0.456044
    clf.fit(features_scaled, labels)

    # 下面注释的这个是利用随机调参，去搞c和gamma的值，不过效果不是很好
    # param_distributions = {
    #     "gamma": reciprocal(0.001, 0.1),
    #     "C": uniform(1, 10)
    # }
    # rnd_search_cv = RandomizedSearchCV(clf, param_distributions, n_iter=10, verbose=2, cv=3)
    # rnd_search_cv.fit(features_scaled, labels)
    # print("rnd_search_cv.best_estimator_:", rnd_search_cv.best_estimator_)
    # print("rnd_search_cv.best_score_:", rnd_search_cv.best_score_)
    # print("rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train):", rnd_search_cv.best_estimator_.fit(features_scaled, labels))

    # 下面的代码是保存模型的
    makedir(model_path)
    joblib.dump(clf, model_path + 'model')
    print("训练之后的模型存放在model文件夹中")

    for feat_path in glob.glob(os.path.join(test_feature_path, '*.feat')):
        total += 1
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        data_test_feat_scaler = scaler.transform(data_test_feat)
        result = clf.predict(data_test_feat_scaler)
        # result = rnd_search_cv.best_estimator_.predict(data_test_feat_scaler)
        if int(result[0]) == int(data_test[-1]):
            correct_number += 1
    rate = float(correct_number) / total
    print("correct_number:", correct_number)
    print("total:", total)
    end_time = time.time()
    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (end_time - start_time))


def extract_feature():
    print("get_feature training is Starting....")
    trian_img_name_label = load_set("train")
    train_img_list, train_img_name_list = get_pic_picname("train")
    get_feature(train_img_list, train_img_name_list, trian_img_name_label, train_feature_path)

    print("get_feature testing is Starting....")
    test_img_name_label = load_set("val")
    test_img_list, test_img_name_list = get_pic_picname("val")
    get_feature(test_img_list, test_img_name_list, test_img_name_label, test_feature_path)


if __name__ == "__main__":
    print("code is Starting....")
    extract_feature()
    train_testmodel()
    print("code is Ending.....")
