import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio

# --------------------------------------------------------------------------------
# seam carving:
# Returns a image and a gif
#
# Inputs(set in the main function):
#           input_image_path:   the path of image to segment.
#           mask_image_path:    the image path is the foreground region annotation of the corresponding input image
#           out_image_path:     the image path of image after cropping
#           gif_path:           the gif path for cropping process
#           gif_frames_path:    the storage path of the frame that generates the gif
#           out_height, out_width: the image size after cropping
#
# --------------------------------------------------------------------------------


class SeamCarving:
    def __init__(self, input_filename, out_height, out_width, mask_filename, output_filename, gif_path, gif_frames_path):
        # 初始化参数
        self.out_height = out_height
        self.out_width = out_width
        self.image_num = 0      # 中间生成的图像的个数
        self.if_turn = False    # 标记是否图像被旋转
        self.frames = []        # 保存生成gif的每一帧图像
        self.gif_frames_path = gif_frames_path   # 用于生成gif的每一帧图片的保存地址

        # 读取图像并存储
        self.input_image = cv2.imread(input_filename).astype(np.float64)

        # 获取输入图像的长宽，shape得到的第三个返回数是通道数，前两个值是长宽
        self.in_height, self.in_width = self.input_image.shape[:2]

        # 读取前景图并存储
        self.mask_image = cv2.imread(mask_filename, 0).astype(np.float64)   # imread的第二个参数为0表示读取的图像是灰度图，channel=1

        # 设置一个初始输出图像
        self.output_image = self.input_image.copy()

        # 开始计算
        self.seam_carving()

        # 保存结果图并在窗口显示
        self.save_image(output_filename, self.output_image)
        # 是按照RGB的形式输出的，但图片保存的是BGR，因此弹出的结果和正常结果不太相同
        plt.imshow(self.output_image.astype(np.uint8))
        plt.show()

        # 生成gif, 在裁剪的过程中已经保存好图像
        imageio.mimsave(gif_path, self.frames, 'GIF', duration=0.1)
        print("The cropping is completed, picture and gif have been generated")
        

        # test
        # 输出能量图
        # e_map = self.get_energy_map()
        # e_map[np.where(self.mask_image > 0)] *= 1000    # 加了掩码之后的能量图
        # plt.imshow(e_map)
        # plt.show()

        # 输出旋转图
        # self.output_image = np.rot90(self.output_image, k=3 )
        # self.save_image("rotate.png",self.output_image)
        # plt.imshow(self.output_image.astype(np.uint8))
        # plt.show()


    def seam_carving(self):
        """
            先裁剪宽，再将图像进行翻转，同样的步骤裁剪宽，反转回来之后实际上是在裁剪高
        """
        # the number of rows and columns needed to be removed
        reduction_row = int(self.in_height - self.out_height)
        reduction_col = int(self.in_width - self.out_width)

        # firstly,remove column
        self.seamcarving_in_one_dir(reduction_col)

        # secondly, remove row
        # 旋转,k=1顺时针旋转90°，k=3，顺时针270°
        self.if_turn = True     # 在存储生成gif的图像时要判断是否将图像恢复后保存
        self.output_image = np.rot90(self.output_image, k=1)
        self.mask_image = np.rot90(self.mask_image, k=1)
        # seamcarving
        self.seamcarving_in_one_dir(reduction_row)
        # 旋转回来
        self.output_image = np.rot90(self.output_image, k=3)
        self.mask_image = np.rot90(self.mask_image, k=3)


    def seamcarving_in_one_dir(self, num_seam):
        """
            在某个方向上进行seam carving 行或列
        """
        for i in range(num_seam):
            energy_map = self.get_energy_map()          # 根据output image获取能量图（经过了每一步的seam carving）
            energy_map[np.where(self.mask_image > 0)] *= 1000   # 根据mask图进行判断，将被标记的前景图灰度值255的地方在能量图上标记为更大的值（保护）
            cumulative_map = self.forward(energy_map)   # 根据forward方法，获取所有最短路径的累计能量图
            seam_id = self.find_seam(cumulative_map)    # 反向查找，获取最短路径的id，为了能删除这个seam
            self.get_image_process(seam_id) # 生成过程图，最后做成gif
            self.delete_seam(seam_id)       # 删除seam
            self.delete_mask_seam(seam_id)  # 删除mask中的seam，要与outout image对应

    def get_energy_map(self):
        '''
            分离BGR通道, 分别求梯度, 最后将三元素的能量加起来, 获得原始能量的能量图
        '''
        b, g, r = self.spilt_BGR(self.output_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))  # x方向和y方向
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))  
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))  
        return b_energy + g_energy + r_energy


    def forward(self, energy_map):
        '''
            根据forward更新梯度, 获得所有可能的“最小”能量累积值
        '''
        # 假设删除第i行j列后的梯度差，当上一行分别位于它的左上方、正上方和右上方
        C_L = self.get_C(0)
        C_U = self.get_C(1)
        C_R = self.get_C(2)

        m, n = energy_map.shape
        M_output = np.copy(energy_map)    # 输出的矩阵m，是能量值的累加
        # 更新能量图，根据这个像素点的梯度以及
        for i_row in range(1,m):    # 从第1行开始，要判断它是上一行的什么
            for j_col in range(n):
                # 不管什么情况，上一行在它的正上方的情况是一定会出现的
                up = M_output[i_row-1, j_col] + C_U[i_row, j_col]   
                # 左边没有其他的列，因此上一行的像素点只能在它的右上方和正上方
                if j_col == 0:
                    right = M_output[i_row-1, j_col+1] + C_R[i_row, j_col]
                    M_output[i_row, j_col] = energy_map[i_row, j_col] + min(right, up)
                # 右边没有其他的列，因此上一行的像素点只能在它的左上方和正上方
                elif j_col == n-1:
                    left = M_output[i_row-1, j_col-1] + C_L[i_row, j_col]
                    M_output[i_row, j_col] = energy_map[i_row, j_col] + min(left, up)
                # 几种情况都会出现
                else:
                    left = M_output[i_row-1, j_col-1] + C_L[i_row, j_col]
                    right = M_output[i_row-1, j_col+1] + C_R[i_row, j_col]
                    M_output[i_row, j_col] = energy_map[i_row, j_col] + min(left, up, right)

        return M_output


    def get_C(self, id):
        '''
            删除第i行j列后的梯度差, 当于上一行分别位于它的左下方、正下方和右下方时的梯度差
            当id = 0时, 计算上一行位于要被删去的像素左上方
            当id = 1时, 计算上一行位于要被删去的像素正上方
            当id = 2时, 计算上一行位于要被删去的像素右上方
        '''
        b, g, r = self.spilt_BGR(self.output_image)
        kernel_1 = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])  # I(i,j + 1) − I(i,j − 1)
        kernel_2 = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])  # I(i − 1,j) − I(i,j − 1)
        kernel_3 = np.array([[0., 1., 0.], [0., 0., -1.], [0., 0., 0.]])  # I(i − 1,j) − I(i,j + 1)
        if id == 0 :
            output = np.absolute(cv2.filter2D(b, -1, kernel_1)) + np.absolute(cv2.filter2D(b, -1, kernel_2)) + \
                     np.absolute(cv2.filter2D(g, -1, kernel_1)) + np.absolute(cv2.filter2D(g, -1, kernel_2)) + \
                     np.absolute(cv2.filter2D(r, -1, kernel_1)) + np.absolute(cv2.filter2D(r, -1, kernel_2))
        elif id == 1:
            output = np.absolute(cv2.filter2D(b, -1, kernel_1)) + \
                     np.absolute(cv2.filter2D(g, -1, kernel_1)) + \
                     np.absolute(cv2.filter2D(r, -1, kernel_1))
        elif id == 2:
            output = np.absolute(cv2.filter2D(b, -1, kernel_1)) + np.absolute(cv2.filter2D(b, -1, kernel_3)) + \
                     np.absolute(cv2.filter2D(g, -1, kernel_1)) + np.absolute(cv2.filter2D(g, -1, kernel_3)) + \
                     np.absolute(cv2.filter2D(r, -1, kernel_1)) + np.absolute(cv2.filter2D(r, -1, kernel_3))
        return output


    def find_seam(self, cumulative_map):
        '''
            利用动态规划的方法进行反向查找, 找到一整条路径
        '''
        m, n = cumulative_map.shape
        path_output = np.zeros((m,), dtype=np.uint32)   # 最后得到的是一条缝的位置id（位于哪一列），m行1列
        path_output[-1] = np.argmin(cumulative_map[-1]) # 最后一行的能量值中最小的坐标，从这个位置开始反向计算
        for i_row in range(m-2, -1, -1):
            temp_id = path_output[i_row + 1]    # 上一步获取到的最小值的id，这一行的最小值id只会在 temp_id-1、temp_id、temp_id+1这三列里
            if temp_id == 0:
                path_output[i_row] = np.argmin(cumulative_map[i_row, :2])
            elif temp_id == n-1:
                path_output[i_row] = temp_id - 1 + np.argmin(cumulative_map[i_row, temp_id - 1 : temp_id + 1])  # 以temp_id - 1为基数，argmin求得最小值的id 0/1，加起来就可以得到真正的列号
            else:
                path_output[i_row] = temp_id - 1 + np.argmin(cumulative_map[i_row, temp_id - 1 : temp_id + 2])  

        return path_output


    def delete_seam(self, seam_id):
        '''
            将输出图像，删除指定的一条缝
        '''
        m, n = self.output_image.shape[:2]  # 输出第0和1个数
        img_output = np.zeros((m, n-1, 3))
        for i_row in range(m):
            del_col = seam_id[i_row]
            img_output[i_row, :, 0] = np.delete(self.output_image[i_row, :, 0], del_col)    # 删除第i行的第del_col列，然后将值赋给新的图
            img_output[i_row, :, 1] = np.delete(self.output_image[i_row, :, 1], del_col)
            img_output[i_row, :, 2] = np.delete(self.output_image[i_row, :, 2], del_col)
        self.output_image = img_output


    def delete_mask_seam(self, seam_id):
        '''
            将mask图像删除对应的一条seam
        '''
        m, n = self.mask_image.shape
        img_output = np.zeros((m, n-1))
        for i_row in range(m):
            del_col = seam_id[i_row]
            img_output[i_row, :] = np.delete(self.mask_image[i_row, :], del_col)
        self.mask_image = img_output

    
    def get_image_process(self, seam_id):
        """
            将得到的路径标记成红色, 表示这一条seam需要被删除
            然后保存这张图片, 并将其存入指定路径
            之后使用, 作为gif中的一帧读入
        """
        m = self.output_image.shape[0]  # 宽度
        img_output = self.output_image.copy()
        for i_row in range(m):
            del_col = seam_id[i_row]
            img_output[i_row, del_col, 0] = 0
            img_output[i_row, del_col, 1] = 0
            img_output[i_row, del_col, 2] = 255
        path = self.gif_frames_path + str(self.image_num) + ".png"
        # 如果图像是被翻转的，要将图像复原后再保存
        if self.if_turn:
            img_output = np.rot90(img_output, k=3)
        self.save_image(path, img_output)
        self.frames.append(imageio.v2.imread(path))
        self.image_num += 1
        
    
    def spilt_BGR(self, image):
        '''
            使用 NumPy 切片得到分离通道, cv2是按照BGR的格式读入的
        '''
        image_ = image.copy()
        b = image_[:, :, 0]
        g = image_[:, :, 1]
        r = image_[:, :, 2]
        return b, g, r


    # 输入图像数据以及路径，保存图像
    def save_image(self, filename, image):
        cv2.imwrite(filename, image.astype(np.uint8))



# 主函数
if __name__ == '__main__':
    image_id = 992
    input_image_path = "../data/imgs/" +str(image_id) + ".png"
    mask_image_path = "../data/gt/" +str(image_id) + ".png"
    out_image_path = "result_" +str(image_id) + ".png"
    gif_path = "gif_" +str(image_id) + ".gif"
    gif_frames_path = "./image_gif/"
    out_height, out_width = 100, 150
    # 裁剪后的图像尺寸示例：
    # 92: 110, 100; 192: 100, 100; 292: 100, 130 ;392: 160, 130; 492: 110, 140 ; 592: 110, 150 
    # 692, 90, 150; 792, 110, 90 ; 892,90, 70; 992: 100, 150
    obj = SeamCarving(input_image_path, out_height, out_width, mask_image_path, out_image_path, gif_path, gif_frames_path)