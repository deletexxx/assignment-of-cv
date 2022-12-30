import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNN
from joblib import dump, load
import cv2
import sys
sys.path.append(r"../2_Segmentation")
import segmentation as seg

# --------------------------------------------------------------------------------
# train a classification model:
#
# generated during training:
#               histogram_data.npy: 1024-dimensional RGB histogram distribution for each region (training data)
#               zone_label_data:    region labels for each region (training data)
#               pca.joblib: for dimensionality reduction models
#               cluster_center.npy: 50 cluster centers obtained after kmeans clustering
#               knn.joblib: classification model obtained after training data
#               
# Print:
#               image_id of the test sample
#               The number of regions into which the test sample is divided
#               IOU
#               1024-dimensional RGB distribution histogram data size
#               20-dimensional data size after dimensionality reduction
#               70-dimensional feature representation data size
#               the accuracy of the regional prediction.
#               
#
# Inputs(set in the main function):
#               if_train: Decide whether to retrain the classification model.
#               Decide whether to recalculate the training data during training, if not, read the calculated data directly
#               test_image_id: image id of the test sample
#               image_path: the path of image to test.
#               mask_path:    the image path is the foreground region annotation of the corresponding input image
#               k: constant for threshold function of test image.
# --------------------------------------------------------------------------------

class GET_DATA:
    def __init__(self, renew_find_data = False, if_read_data = True):
        # 数据的保存
        self.histogram_path = "./histogram_data.npy"
        self.zone_label_path = "./zone_label_data.npy"
        # 读取图像数据的路径
        self.image_path = "../source_data/imgs/"
        self.mask_image_path = "../source_data/gt/"
        # 1024维颜色区域对比度特征
        self.image_his_all = []
        # 区域对应的前景标记
        self.image_zone_label = []

        # 是否重新计算训练数据
        if renew_find_data:
            print("Start recomputing training data")
            self.getRGBhis()

        # 是否读取训练数据
        if if_read_data:
            print("read training data")
            self.read_data()
    
    # 读取数据
    def read_data(self):
        self.image_his_all = np.load(self.histogram_path)
        self.image_zone_label = np.load(self.zone_label_path)
        print("The size of the read matrix is " + str(self.image_his_all.shape))
    
    # 获取随机选取的200张图片的颜色归一化直方图
    def getRGBhis(self):
        num_sample = 0  # 要求划分的区域为50-70，200张图很难每个去调试k，因此遍历所有样本，将划分在50-70个区域的样本作为巽离岸样本
        image_id = 0
        while num_sample!= 200 and image_id != 1000:
            print("image "+str(image_id)+" :")
            path = self.image_path + str(image_id+1) + ".png"   # 图像路径
            mask_path = self.mask_image_path + str(image_id+1) + ".png"     # mask图像路径
            # k = 200, min_pix=50
            # 类的一个实例化，对image进行区域划分
            image_seg = seg.Segmentation(path, 200, 50, mask_path)  # 类的一个实例化，对image进行区域划分
            root, zone_label, node_parent = image_seg.get_data_comput_his()

            zone_num = len(zone_label)
            # 判断划分的区域数量是否在满足的范围内
            if zone_num>=50 and zone_num<=70:
                # 在范围内就再进行获取数据的操作
                # 获取整个图片的RGB分布直方图
                image = cv2.imread(path).astype(np.float64)
                his = self.get_RGB_Histogram(image)
                # 获取区域划分的直方图并进行拼接
                zone_his = self.get_zone_RGB_Histogram(image, root, node_parent, his)
                # 判断是要进行区域和区域拼接成矩阵还是直接加入
                if num_sample == 0:
                    self.image_his_all = zone_his.copy()
                    self.image_zone_label = zone_label.copy()
                else:
                    self.image_his_all = np.vstack((self.image_his_all, zone_his))
                    self.image_zone_label = self.image_zone_label + zone_label
                num_sample += 1 # 符合条件，被加入训练样本
                print("it is a sample "+str(num_sample))

            image_id += 1   # 遍历下一张图

        np.save(self.histogram_path, self.image_his_all)
        self.image_zone_label = np.array(self.image_zone_label) # 将list转化为矩阵方便保存
        np.save(self.zone_label_path, self.image_zone_label)


    # 计算图片的归一化RGB直方图,要求是8*8*8，因此将颜色值域256划分成8个区域
    def get_RGB_Histogram(self, img):
        histogram = np.zeros((8,8,8))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                B = int(img[i][j][0]/32)
                G = int(img[i][j][1]/32)
                R = int(img[i][j][2]/32)
                histogram[B, G, R] += 1
        # 归一化，预防一些位置和是0的情况
        histogram = np.divide(histogram, np.sum(histogram), out = np.zeros_like(histogram,dtype=np.float64), where= np.sum(histogram))
        # histogram /= np.sum(histogram)  # 归一化
        # print(np.count_nonzero(histogram))  # test
        return histogram.reshape(-1)    # 转换为1维后输出
    
    def get_zone_RGB_Histogram(self, img, root, node_parent, img_his):
        zone_num = len(root)    # 区域的个数，只有当数据满足要求之后才会被计算、返回
        zone_his = np.zeros([zone_num, 8*8*8*2])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                root_vertex = self.find_root(i*img.shape[1]+j, node_parent)  # 获取根节点，确定区域的位置
                root_index = root.index(root_vertex)        # 获取根节点的下标
                B = int(img[i][j][0]/32)
                G = int(img[i][j][1]/32)
                R = int(img[i][j][2]/32)
                zone_his[root_index, (B*8+G)*8+R] += 1
        # 进行区域的归一化，每一行除以它的和
        sum_row = np.sum(zone_his, axis=1)# 按行求和
        for i in range(zone_num):
            zone_his[i, :512] = np.divide(zone_his[i, :512], sum_row[i], out = np.zeros_like(zone_his[i, :512],dtype=np.float64), where= sum_row[i])
            # zone_his[i, :512] /= sum_row[i] # 前512个保存的是区域RGB直方图
            zone_his[i, 512:] = img_his.copy()  # 后512个元素是原图的RGB直方图，即将区域直方图域原图直方图进行拼接形成1024维
        print("histogram size" + str(zone_his.shape))
        return zone_his
    
    # 查找根节点
    def find_root(self, vertex, node_parent):
        if node_parent[vertex] == vertex:
            return vertex        
        return self.find_root(node_parent[vertex], node_parent)

    def return_data(self):
        return self.image_his_all, self.image_zone_label



class train:
    def __init__(self, if_renew_get_data = False, if_train = False):
        self.if_renew_get_data = if_renew_get_data
        self.cluster_center_path = "./cluster_center.npy"
        if if_train:
            self.train_model()
        else:
            self.pca = load("pca.joblib")
            self.knn = load("knn.joblib")
            self.cluster_center = np.load(self.cluster_center_path)

        # 如果直接进行测试，需要保存knn、pca模型以及50个聚类中心


    def train_model(self):
        # 获取用于训练的图像数据
        DATA = GET_DATA(self.if_renew_get_data)
        self.image_his_all, self.image_zone_label = DATA.return_data()
        self.cluster_center = []

        # 降维
        self.dim_reduc()

        # 聚类
        self.k_means()

        # 计算50个样本中心的每一个中心和样本的点积/余弦相似度
        cos_dis = self.comp_dis()

        # 将降维后的20维数据与得到的与50个聚类中心的特征进行拼接，70维
        self.feature_rep = np.hstack((self.image_his_all, cos_dis)) # 70维特征表示
        # print(self.feature_rep.shape)

        # 使用KNN算法进行分类        
        self.knn = KNN()
        self.knn.fit(self.feature_rep, self.image_zone_label)
        dump(self.knn, "knn.joblib")

    
    def comp_dis(self):
        dot_product = np.dot(self.image_his_all, self.cluster_center.T) # num*50, num个样本和50个中心分别的点积
        vector_length_sample = np.linalg.norm(self.image_his_all, axis=1).reshape(-1, 1)    # 1列
        vector_length_center = np.linalg.norm(self.cluster_center, axis=1).reshape(1, -1)   # 1行

        cos_dis = dot_product / np.dot(vector_length_sample, vector_length_center)

        return cos_dis


    
    def k_means(self):
        kmeans = KMeans(n_clusters = 50)    # 50个聚类中心
        kmeans.fit(self.image_his_all)      # 进行kmeans聚类
        self.cluster_center = kmeans.cluster_centers_.copy() # 获取聚类中心
        np.save(self.cluster_center_path, self.cluster_center)
        # print(cluster_center.shape)

    # 利用PCA进行降维
    def dim_reduc(self):
        self.pca = PCA(n_components=20)
        dump(self.pca, "pca.joblib")
        self.image_his_all = self.pca.fit_transform(self.image_his_all)
        print("After dimensionality reduction using PCA, the size of the matrix is " + str(self.image_his_all.shape))

    # 进行测试
    def test_image(self, image_path, mask_image_path, image_id, k):
        # 设置结果图像的保存路径
        predict_image_path = "./predict_" + str(image_id) + ".png"  # 预测区域前景图
        # segment_image_path = "./segmentation_" + str(image_id) + ".png" # 区域划分图，不保存了，之前第2题中得到过
        mask_seg_path = "./mask_seg_" + str(image_id) + ".png" # 区域划分标记图

        # 实例化read data类
        get_test_data = GET_DATA(renew_find_data = False, if_read_data = False) # 只是用来获取一张图片的RGB区域直方图，不必读取数据
        image = cv2.imread(image_path).astype(np.float64)

        # 获取图像的RGB直方图
        histogram = get_test_data.get_RGB_Histogram(image)
        # 获取划分后每个区域的RGB直方图
        # 先进行区域划分
        image_seg = seg.Segmentation(image_path, k, min_piexl=50, mask_filename=mask_image_path, if_computer_IOU=True,  output_filename=None,zone_maker_image_filename=mask_seg_path)
        root, test_label, node_parent = image_seg.get_data_comput_his()
        zone_histogram = get_test_data.get_zone_RGB_Histogram(image, root, node_parent, histogram)

        # 利用模型进行降维
        zone_histogram = self.pca.fit_transform(zone_histogram)
        print("after PCA dimensionality reduction, the matrix size " + str(zone_histogram.shape))

        # 同样计算点积相似度
        dot_product = np.dot(zone_histogram, self.cluster_center.T) # num*50, num个样本和50个中心分别的点积
        vector_length_sample = np.linalg.norm(zone_histogram, axis=1).reshape(-1, 1)    # 1列
        vector_length_center = np.linalg.norm(self.cluster_center, axis=1).reshape(1, -1)   # 1行
        cos_dis = dot_product / np.dot(vector_length_sample, vector_length_center)

        # 特征拼接
        test_feature_rep = np.hstack((zone_histogram, cos_dis)) # 70维特征表示
        print("feature representation " + str(test_feature_rep.shape))

        # 用训练好的knn模型预测结果
        predict_label = self.knn.predict(test_feature_rep)
        right_num = np.sum(predict_label == test_label)
        print("The accuracy of the regional prediction is " + str(right_num/len(test_label)))

        # 将预测的标签进行绘制
        predict_image = self.draw_zone_maker_image(predict_label, image_seg, image.shape[0], image.shape[1], get_test_data) 
        self.save_image(predict_image_path, predict_image)


    # 绘制区域标记好的图像，前景区域记为0，背景255
    def draw_zone_maker_image(self, label, image_seg, height, width, get_test_data):
        root, zone_label, node_parent = image_seg.get_data_comput_his()
        image = np.zeros((height,width)) # 灰度图，不必有RGB三通道
        for i_row in range(height):
            for j_col in range(width):
                v_root = get_test_data.find_root(i_row*width + j_col, node_parent)  # 调用GET_DATA类里的查找根节点的函数
                v_root_id = root.index(v_root)
                if label[v_root_id]:
                    image[i_row, j_col] = 255
                else:
                    image[i_row, j_col] = 0     
        return image

    # 保存图像
    def save_image(self, filename, image):
        cv2.imwrite(filename, image.astype(np.uint8))


        
        

if __name__ == '__main__':
    # 可以决定是否要重新读取数据，也可以决定是否要训练模型，不进行训练就直接读取已经训练好的模型
    if_train = False
    if_renew_get_data = False
    mode = train(if_renew_get_data, if_train)
    test_image_id = 92
    image_path = "../source_data/imgs/" + str(test_image_id) + ".png"
    mask_image_path = "../source_data/gt/" + str(test_image_id) + ".png"
    k = 200
    # 592 k100； 792，100
    print("test image_id " + str(test_image_id))
    mode.test_image(image_path, mask_image_path, test_image_id, k)# 892最好