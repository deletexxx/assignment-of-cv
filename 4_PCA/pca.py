import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2

# --------------------------------------------------------------------------------
# PCA:
#       
# generate:
#               Heatmap of the 16 most important eigenvectors
#               image after dimensionality reduction   
#
# Print:
#               From m dimensions, down to k dimensions
#               Original data size
#               Compressed data size
#               Restored data size
#               
#
# Inputs(set in the main function):
#               image_id: image_id of the test sample
#               in_path: the path of test image for dimensionality reduction.
#               out_path: image path after dimensionality reduction
#               dim: preserved dimension
# --------------------------------------------------------------------------------

class PCA:
    def __init__(self, input_filename, dim, output_filename):
        self.out_image_path = output_filename

        # 按照灰度图进行读取
        self.in_image = cv2.imread(input_filename, 0).astype(np.float64)
        self.height, self.width = self.in_image.shape[:2]
        self.dim = dim
        
        # 裁剪图像，成12的倍数
        self.block_m = int(self.height/12)
        self.block_n = int(self.width/12)
        self.num_data = self.block_m*self.block_n   # 数据的个数，之后也会表示为行数
        self.in_image = self.in_image[0:self.block_m*12, 0:self.block_n*12]
        self.height, self.width = self.in_image.shape[:2]
        
        # 存储转换后的数据，12*12行，num_data列，一列表示一个数据
        self.data = np.zeros([12*12, self.num_data])

        # 输出的图像
        self.out_image = np.zeros([self.height, self.width])
        
        self.PCA_step()


    def PCA_step(self):
        self.div_image()    # 划分区域，每12*12的一个区域作为一行显示，共有block_m*block_n行12*12维数据
        # self.show_block()   # 显示划分的区域

        # 去中心化
        decent_data = self.data.copy()  # 去中心化后的数据
        self.decentralization(decent_data)

        # SVD奇异值分解
        U, s = np.linalg.svd(decent_data)[:2]   # 只需要XX^T的特征向量即可


        # 奇异值本身就从大到小排列
        # 取前k列，作为主成分，P是转置
        P = U[:, :dim].T

        # 生成前 16 个最主要的特征向量的可视化图
        self.Eigenvector_Visualization_Plot(U.T, s)

        # 将源数据进行降维
        print("从144维,  降到"+str(self.dim)+"维")
        print("原本的数据大小: "+str(self.data.shape))  # 原本数据大小
        self.data =P.dot(self.data)  # 得到正交向量后压缩源数据
        print("压缩后数据大小: " + str(self.data.shape))  # 压缩后数据大小

        # 复原数据
        self.data = P.T.dot(self.data)
        print("复原后数据大小: " + str(self.data.shape))  # 复原后数据大小

        # 将用矩阵表示的图像数据进行复原
        self.restore_image()

        # 展示生成的图像
        plt.imshow(self.out_image.astype(np.uint8), cmap='gray')
        plt.show()

        # 保存降维后的又复原的图像
        self.save_image(self.out_image_path, self.out_image)

        # 利用协方差矩阵的特征向量进行PCA降维
        # 求协方差矩阵, （样本的）
        # C = self.data.dot(self.data.T)/self.num_data    # 144*144
        # eigenvalue, featurevector = np.linalg.eig(C)
        # print(featurevector.shape)

        # # 将特征值排序，保存排序后的下标
        # sorted_indices = np.argsort(eigenvalue)[::-1]

        # # 取前k列，作为主成分
        # P = np.zeros([144, self.dim], dtype=complex)    # 获得的特征向量中有虚数，构造0复数矩阵
        # for i in range(self.dim):
        #     P[:, i] = featurevector[:, sorted_indices[i]]
        
        # # 求降到dim维后的数据
        # self.data = P.T.dot(self.data)
        # # print(self.data.shape)

        # # 复原数据
        # self.data = P.dot(self.data)
        # self.show_block()
        # # print(self.data.shape)


    # 去中心化
    def decentralization(self, decent_data):
        aver_row = np.average(decent_data, axis = 1) # 每一行的均值
        # 每一行减去均值，相当于所有数据每一维去中心
        for i in range(12*12):
            decent_data[i, :] = decent_data[i, :] - aver_row[i]

    # 划分图像
    def div_image(self):
        for i in range(self.block_m):
            for j in range(self.block_n):
                for k in range (12):
                    self.data[k*12:(k+1)*12, i*self.block_n+j] = self.in_image[i*12+k, j*12:(j+1)*12]
                    # 前半段是指新矩阵的第i*self.block_n+j列，第k*12列到第(k+1)*12行
                    # 后半段是指原图像矩阵的i*12+k行，到第j*12:(j+1)*12列
    
    # 生成划分区域后的图像
    def show_block(self):
        for i in range(self.block_m):
            for j in range(self.block_n):
                block_image = self.data[:, i*self.block_n+j]    # 每一列表示一个block
                block_image = block_image.reshape((12,12))      # 获取一列后重新生成图像的形式
                plt.subplot(self.block_m, self.block_n, i*self.block_n+j+1)     # 生成子图
                plt.imshow(block_image.astype(np.uint8), cmap='gray')           # 作为灰度图像显示
                plt.axis('off')
        plt.show()


    # 将图像复原
    def restore_image(self):
        for i in range(self.block_m):
            for j in range(self.block_n):
                for k in range (12):
                    self.out_image[i*12+k, j*12:(j+1)*12] = self.data[k*12:(k+1)*12, i*self.block_n+j]


    # 获取特征值最大的前16个特征向量的热力图
    def Eigenvector_Visualization_Plot(self, vec, s):
        for i in range(16):
            eigenvector = vec[i, :]
            eigenvalue = s[i]
            data = eigenvector.reshape(12,12)
        
            ax = sns.heatmap(data, cmap='gray', cbar=False, square=True) # 用灰度图表示，彩色的也行
            ax.set_title("eigenvalue = " + str(eigenvalue) + ", Heatmap of eigenvector")  # 图标题
            figure = ax.get_figure()
            figure.savefig("heatmap"+str(i)+".png")
        # plt.show()
        
        
    
    # 保存图像
    def save_image(self, filename, image):
        cv2.imwrite(filename, image.astype(np.uint8))


# 主函数
if __name__ == '__main__':
    image_id = 992
    in_image_path = "../data/imgs/"+str(image_id)+".png"
    dim = 60
    out_image_path = str(image_id) + "result_" + str(dim) + "D.png"
    PCA(in_image_path, dim, out_image_path)