import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

# 一定要在代码所在目录下运行，否则找不到输入图像的位置
# --------------------------------------------------------------------------------
# segmentation:
# Returns a image: result plot with different colors for different regions
#
# Inputs(set in the main function):
#           in_image_path:      the path of image to segment.
#           mask_image_path:    the image path is the foreground region annotation of the corresponding input image
#           out_image_path:     the image path of image after cropping
#           zone_maker_image_path: The storage path of the generated foreground image
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
# 
# Print:
#           num_zone: number of connected components in the segmentation.
#           IOU
# --------------------------------------------------------------------------------


class graph_vertex_operate:
    def __init__(self, num_node):
        # 初始化
        self.num_node = num_node
        self.parent = [i for i in range(num_node)]  # 存储每个节点的父节点
        self.rank = [0 for i in range(num_node)]    # 级别，用来判断谁做根节点
        self.size = [1 for i in range(num_node)]    # 每个节点以它为根节点，子树上节点的个数
        self.all_root = []  # 所有根节点的集合
        self.prospect_node_num = [] # 每个根节点下区域里在mask前景图中对应的像素个数
        self.if_prospect_zone = []  # 判断这个根节点下的区域是否是前景图
        self.if_prospect_zone_label = []    # 是否是前景图的标签，只是把true写成1，false写成0，3题中用
    
    # 查找根节点
    def find_root(self, vertex):
        if self.parent[vertex] == vertex:
            return vertex
        
        return self.find_root(self.parent[vertex])
    
    # 将两个节点所在区域合并，共用一个根节点的过程
    def merge(self, vertex1, vertex2):
        v1_root = self.find_root(vertex1)
        v2_root = self.find_root(vertex2)

        if v1_root != v2_root:
            # 让v1的根节点作为父节点
            if self.rank[v2_root] > self.rank[v1_root]:
                v1_root, v2_root = v2_root, v1_root

            # 更新父节点
            self.parent[v2_root] = v1_root
            # 更新子树节点个数
            self.size[v1_root] += self.size[v2_root]

            # 级别相等 给父节点的rank+1，父节点的rank更高
            if self.rank[v1_root] == self.rank[v2_root]:
                self.rank[v1_root] += 1
    
    # 返回子树节点个数
    def sub_node_size(self, vertex):
        return self.size[vertex]

    # 获取所有的根节点，保存在一个list中方便查找
    def get_all_root(self):
        self.all_root = []  # 以免重复调用不断append导致结果出错
        for vertex in range(self.num_node):
            if vertex == self.find_root(vertex):
                self.all_root.append(vertex)
        return self.all_root
    
    # 返回根节点个数
    def root_num(self):
        return len(self.all_root)

    # 返回根节点代表的区域是否是前景图，按照list存储根节点的顺序存储
    def get_zone_label(self):
        return self.if_prospect_zone_label

    # 返回所有节点的父节点，来查找根节点
    def node_parent(self):
        return self.parent

    # 根据父节点，进行区域标记，将分割区域进行标记
    def zone_maker(self, mask_image, height, width):
        self.get_all_root() # 计算得到根节点
        self.prospect_node_num = [0 for i in range(self.root_num())]
        # 在每个根节点上计数，该区域在前景中的数量，然后判断它是前景还是背景，然后标记并输出，计算IOU
        for i_row in range(height):
            for j_col in range(width):
                if mask_image[i_row, j_col] > 0:
                    self.add_prospect_node_num(i_row*width + j_col)
        # 判断各个根节点代表的区域是否是前景区域
        self.judge_prospect_zone()
        # 绘图，得到我们划分的前景区域，并且计算这个图中前景区域的大小
        zone_maker_image = self.draw_zone_maker_image(height, width) 
        return zone_maker_image

    # 该节点的根节点上计算区域的前景数
    def add_prospect_node_num(self, vertex):
        # 查找该节点的根节点
        vertex_root = self.find_root(vertex)
        # 查找该根节点在根节点的列表中的下标，这个下标也对应该根节点对应区域的前景像素个数
        vertex_root_id = self.all_root.index(vertex_root)
        self.prospect_node_num[vertex_root_id] += 1

    # 判断所有根节点所在的区域是否是前景
    def judge_prospect_zone(self):
        for root_i in range(self.root_num()):
            # 获取该区域节点个数
            zone_pix = self.sub_node_size(self.all_root[root_i])
            # 前景元素占0.5以上，认为是前景图，否则是背景图
            if self.prospect_node_num[root_i]/zone_pix > 0.5:
                self.if_prospect_zone.append(True)
                self.if_prospect_zone_label.append(1)
            else:
                self.if_prospect_zone.append(False)
                self.if_prospect_zone_label.append(0)

    # 绘制区域标记好的图像，前景区域记为0，背景255
    def draw_zone_maker_image(self, height, width):
        image = np.zeros((height,width)) # 灰度图，不必有RGB三通道
        for i_row in range(height):
            for j_col in range(width):
                v_root = self.find_root(i_row*width + j_col)
                v_root_id = self.all_root.index(v_root)
                if self.if_prospect_zone[v_root_id]:
                    image[i_row, j_col] = 255
                else:
                    image[i_row, j_col] = 0     
        return image


class Segmentation:
    def __init__(self, input_filename, k, min_piexl, mask_filename, if_computer_IOU = False, output_filename = None, zone_maker_image_filename = None):
        # 初始化参数
        self.k = k
        self.min_piexl = min_piexl
        self.graph = [] # 包括所有的边

        # 读取图像并存储
        self.input_image = cv2.imread(input_filename).astype(np.float64)
        # 获取输入图像的长宽
        self.height, self.width = self.input_image.shape[:2]

        # 构建集合（标记所有点的父节点，同一个root的节点属于同一个集合）
        self.num_node = self.width*self.height
        self.seg_set = graph_vertex_operate(self.num_node)

        # 初始输出图像，全为0
        self.output_image = np.zeros((self.height, self.width, 3))

        # 开始计算
        self.segmentation()

        # 保存结果图并在窗口显示
        if output_filename is not None:
            self.save_image(output_filename, self.output_image)
            # plt.imshow(self.output_image.astype(np.uint8))
            # plt.show()
        
        # 读取mask图像
        self.mask_image = cv2.imread(mask_filename, 0).astype(np.float64)

        # 先给每个根节点上计数，在前景中的数量，然后判断它是前景还是背景，然后标记并输出，再计算IOU
        zone_maker_image = self.seg_set.zone_maker(self.mask_image, self.height, self.width)    # 得到区域标记好的灰度图
        
        # 计算IOU比例
        if if_computer_IOU:
            IOU = self.compute_IOU(zone_maker_image)    #计算IOU

            # 输出IOU，保存并在窗口显示区域标记好的图
            print("IOU = " + str(IOU))
            self.save_image(zone_maker_image_filename, zone_maker_image)
            # plt.imshow(zone_maker_image.astype(np.uint8))
            # plt.show()

    def segmentation(self):
        # 对图像进行高斯模糊,减少图像噪声以及降低细节层次, 最后的参数sigma可以调参，但在本次实验中不是重点
        self.input_image = cv2.GaussianBlur(self.input_image, (5,5), 0.5)
        
        # 建立图
        self.build_graph()

        # 根据获得的所有边的RGB距离进行排序
        weight = lambda edge: edge[2]   # 每条边根据第三个值，边的权重进行排序
        self.graph = sorted(self.graph, key=weight) 

        # 进行图像分割
        self.segment()
        self.merge_min_set()

        # 进行绘图
        self.get_output_image()


    # 图像分割算法关键区域
    def segment(self):
        MInt_C = [self.get_threshold_based(1)] * self.num_node

        for edge in self.graph: # 从已经排过序的边集去遍历
            # 获取这条边上两个点的根，判断他们是否属于一个区域
            v1_root = self.seg_set.find_root(edge[0])
            v2_root = self.seg_set.find_root(edge[1])
            dis_RGB = edge[2]

            # 如果这条边上两个点不在一个区域内，
            # 那么它是连接两个区域的最短边（按距离升序进行遍历）
            if v1_root != v2_root:
                # 判断他们是否能合并,if(diff(C1,C2)<mint(C1,C2))
                if dis_RGB <= min(MInt_C[v1_root], MInt_C[v2_root]):
                    # 进行合并
                    self.seg_set.merge(v1_root, v2_root)
                    # 更新Mint_C
                    root_now = self.seg_set.find_root(v1_root)
                    # 由于未被遍历的边越小越快被加入某个区域，因此这条边刚被并入某个区域时一定是此时这个区域内的最大距离
                    # 因为和每个区域比较，可以确定的是根节点，因此可以在根节点上保存Int(C)+t(C)的值
                    MInt_C[root_now] = dis_RGB + self.get_threshold_based(self.seg_set.sub_node_size(root_now))

    
    # 将任意可以连通的两个区域进行判断，如果没有达到要求的大小，就将它和最近的区域进行合并
    def merge_min_set(self):
        for edge in self.graph:
            v1_root = self.seg_set.find_root(edge[0])
            v2_root = self.seg_set.find_root(edge[1])

            # 说明并未这两个区域还合并（从最短的边开始遍历），判断两点的区域是否符合每个区域的大小要求
            if v1_root != v2_root:
                if min( self.seg_set.sub_node_size(v1_root), self.seg_set.sub_node_size(v2_root) ) < self.min_piexl:
                    self.seg_set.merge(v1_root, v2_root)
        
                
    # 计算阈值
    def get_threshold_based(self, set_num_node):
        return self.k/set_num_node

    # 创建图，保存所有的边
    def build_graph(self):
        for i_row in range(self.height):
            for j_col in range(self.width):
                if i_row > 0 and j_col < self.width - 1:    
                    self.graph.append(self.get_edge(i_row, j_col, i_row - 1, j_col + 1))    # 计算该点右上的边，否则会越界
                if j_col < self.width - 1:  
                    self.graph.append(self.get_edge(i_row, j_col, i_row, j_col + 1))        # 右边
                if j_col < self.width - 1 and i_row < self.height - 1:  
                    self.graph.append(self.get_edge(i_row, j_col, i_row + 1, j_col + 1))    # 右下
                if i_row < self.height - 1: 
                    self.graph.append(self.get_edge(i_row, j_col, i_row + 1, j_col))        # 正下方

    # 获取节点以及边的权重（RGB距离）
    def get_edge(self, v1_row, v1_col, v2_row, v2_col):
        v1_id = v1_row*self.width + v1_col
        v2_id = v2_row*self.width + v2_col
        # get diff RGB
        b_diff = self.input_image[v1_row, v1_col, 0] - self.input_image[v2_row, v2_col, 0]
        g_diff = self.input_image[v1_row, v1_col, 1] - self.input_image[v2_row, v2_col, 1]
        r_diff = self.input_image[v1_row, v1_col, 2] - self.input_image[v2_row, v2_col, 2]
        distance_RGB = math.sqrt(b_diff**2 + g_diff**2 + r_diff**2)
        return v1_id, v2_id, distance_RGB

    # 绘制输出的图像，将不同区域标记成不同的颜色
    def get_output_image(self):
        root = self.seg_set.get_all_root()   # 得到图中所有的根节点,并为这个根节点生成一个颜色
        root_num = self.seg_set.root_num()
        color = []

        for i in range(root_num):
                # 生成一个随机颜色
                rand_color = [np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)]
                color.append(rand_color)

        print("The image is divided into " + str(root_num) + " regions")
        
        for i_row in range(self.height):
            for j_col in range(self.width):
                vertex_root = self.seg_set.find_root(i_row*self.width + j_col)
                color_id = root.index(vertex_root)
                self.output_image[i_row, j_col] = color[color_id]

    # 计算IOU
    def compute_IOU(self, zone_maker_image):
        intersection_pix_num = 0
        union_pix_num = 0
        for i_row in range(self.height):
            for j_col in range(self.width):
                if self.mask_image[i_row, j_col] > 0 or zone_maker_image[i_row, j_col] > 0:     # 并集
                    union_pix_num += 1
                if self.mask_image[i_row, j_col] > 0 and zone_maker_image[i_row, j_col] > 0:    # 交集
                    intersection_pix_num += 1
        
        IOU = intersection_pix_num/union_pix_num
        return IOU


    # 使用 NumPy 切片得到分离通道
    def spilt_BGR(self, image):
        image_ = image.copy()
        b = image_[:, :, 0]
        g = image_[:, :, 1]
        r = image_[:, :, 2]
        return b, g, r

    # 保存图像
    def save_image(self, filename, image):
        cv2.imwrite(filename, image.astype(np.uint8))

    # 返回一些计算区域直方图需要的数据
    def get_data_comput_his(self):
        root = self.seg_set.get_all_root()
        zone_label = self.seg_set.get_zone_label()
        node_parent = self.seg_set.node_parent()
        return root, zone_label, node_parent



# 主函数
if __name__ == '__main__':
    image_id = 992
    in_image_path = "../data/imgs/" + str(image_id) + ".png"
    mask_image_path = "../data/gt/" + str(image_id) + ".png"
    out_image_path = "result_" + str(image_id) + ".png"
    zone_maker_image_path = "result_marker_" + str(image_id) + ".png"
    k = 350 # k越大划分出来的区域越小192，k=100；392，k=200（效果很好iou=0.91）
    min_pixel = 50
    # 92, 400;  192, 100;292, 120; 392, 200; 492, 120; 592, 200; 692, 150;792, 350;892, 180; 992, 350
    print("image_id "+ str(image_id)) 
    Segmentation(in_image_path, k, min_pixel, mask_image_path, if_computer_IOU=True, output_filename = out_image_path, zone_maker_image_filename = zone_maker_image_path)