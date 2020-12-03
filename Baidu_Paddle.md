## Code1 

```python
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
import numpy as np
np.set_printoptions(precision = 2 )

class BasicModel(fluid.dygraph.Layer):
	def __init__(self, num_classed =59 ):
		super(BasicModel, self).__init__()
		self.pool = Pool2D(pool_size=2,pool_stride=2)
        self.conv = Conv2D(num_channles=3,num_filters=num_classes,filter_size=1)
        
        
	def forward(self,inputs):
		x = self.pool(inputs)
        x = fluid.layers.interpolate(x,out_shape=inputs.shape[2::])
        x = self.conv(x)
         
        
		return x

def main():
	place = paddle.fluid.CPUPlace()
	#palce =  paddle.fluid.CUDAPlace(0)
	with fluid.dygraph.guard(place):
		model = BasicModel(num_classed=59)
		model.eval() #model.train()
		input_data =np.random.rand(1,3,8,8).astype(np.float32)
        print('Input data shape:', input_data.shape)
        input_data = to_variable(input_data)
        output_data = model(input_data)
        output_data = output_data.numpy()
        print('Output data shape:', output_data.shape)
 if __name__="__main__":
    main()
		
```



## Coding 2 demo2

```python
import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid


class Transform(object):
    def __init__(self, size =256):
    	self.size =size
    def __call(self, input, label):
        input = cv2.resize(input, (self.size, self.size),interpolation= cv2.INTER_LINEAR)
        label = cv2.resize(input, (self.size,self.size),interpolation =cv2.INTER_NEARST)
        
        return input, label
    
    
class BasicDataLoader(object):
    def __init__(self,
                image_folder,
                image_list_file,
                transform = None,
                shuffle = True):
    	self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform=transform
        self.shuffle =shuffle
        
        self.data_list = self.read_list()
        
    def read_list(self):
        data_list = []
        with open(self.image_list_file) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder,line.split()[0]
                label_path = os.path.join(self.image_folder,line.split()[1]
				data_list.appened((data_path, label_path))
                                          
        random.shuffle(data_list)                                  
		return data_list
                                          
    def preprocess(self, data, label)                                     h, w, c = data.shape
            h_gt, w_gt = label.shape
            asset h == h_gt, "Error"
            asset w == w_gt, "Error"
                                          
            if self.transform:
               data, label =self.transform(data, label)
                                          
             label = label[:,:,np.newaxis]
                                          
                                     
                                         
	def __len__(self):
    	return len(self.data_list)
                                          
    def __call__(self):
        for data_path, label_path in self.data_list:
        	data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
           	label = cv2.imread(label_path)
            print(data.shape, label.shape)
            data, label = self.preprocess(data, label)
            
            yield data, label
                                          
                                          
def main():
    batch_size = 5
    place = fluid.CPUPlace(0)
    with fluid.dygraph.guard(place):
        transform =Transform(256)
        # create BasicDataloader instance
        basic_dataloader = BasicDataLoader(
        	image_folder = './dummy_data',
            image_list_file = './dummy_data/list.txt',
            transform = transform,
            shuffle = True
        	)
        
        # create fluid.io.DataLoader instance
        dataloader = fluid.io.Dataloader.from_generator(capacity=1,use_multiprocess=False)
        # set sample generator for fluid dataloader
      	dataloader.set_sample_generator(basic_dataloader,
                                       batch_size=batch_size,
                                       places=place)  
        
        num_epoch = 2
        for epoch in range(1,num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx,(data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape},Label shape: {label.shape}')
                
if __name__ == "__main__":
    main()
```





## FCN 网络

### why does FCN work？

FCN　= Fully Convolutional Networks

-什么是Fully Convolutionnal?

​	简而言之：全卷积，没有FC

-如何做语义分割？

 	语义分割  ~= 像素级分类

-和图像分类有什么关系

​	替换FC,换成Conv

经典CNN 模型－ＶＧＧ

![image-20201126194121270](F:\Files\Typora_ImageStore\image-20201126194121270.png)

![image-20201126194422076](F:\Files\Typora_ImageStore\image-20201126194422076.png)



如何处理FC层？

　FC层　－＞　１ｘ１　Conv　　　

​	代表　输入的长和宽不变，把channel改成和卷积channel个数相同，降维／升维　把其转换为需要的分类数。优势：无需考虑输入尺寸有多大



需要Feature map 尺寸变大 

卷积：越卷越小，无法做分割

Upsample:越变越大

​	1.Up-sampling  2.Transpose Conv 3. Un-pooling

1. Up sampling:Bilinear Interpolation 双线性插值

![image-20201126220558909](F:\Files\Typora_ImageStore\image-20201126220558909.png)

![image-20201126220612657](F:\Files\Typora_ImageStore\image-20201126220612657.png)

计算过程：

![image-20201126220842926](F:\Files\Typora_ImageStore\image-20201126220842926.png)

离哪个像素点近一点就加权大一点

代码中看到Up sampling就要知道需要把feature map放大

2. Un-pooling 

   Pooling 的反向操作！ 需要一个INdices，需要告诉像素填回的位置

3. Transpose Conv

    ![image-20201127101025098](F:\Files\Typora_ImageStore\image-20201127101025098.png)

 做一个反卷积，使Feature map变大 ，具体操作

![image-20201127101301772](F:\Files\Typora_ImageStore\image-20201127101301772.png)

首先把kernel做一个180的转换，然后做一个Conv+Padding,

其中Input不变，Kernel水平滑动，然后在相应叠加的位置进行运算得到扩大后的output

![image-20201127101840485](F:\Files\Typora_ImageStore\image-20201127101840485.png)

滑窗对于运算太慢，所以大致是要实现一个矩阵相乘，提高速度，具体就是把Kernel或Input转换成矩阵形式，得到一个output。实际过程中，各种框架实现方式不同，具体形式不同，主要是进行input的转换，因为kernel是我们的学习目标

![image-20201127102337927](F:\Files\Typora_ImageStore\image-20201127102337927.png)

实际可能是这样的过程，然后把output按序存储成矩阵



![image-20201127102713542](F:\Files\Typora_ImageStore\image-20201127102713542.png)

这三种方法就是在❓处把Feature map变大,方便分割

### FCN-Coding

网络结构

<img src="F:\Files\Typora_ImageStore\image-20201127103247418.png" alt="image-20201127103247418" style="zoom:150%;" />



<img src="F:\Files\Typora_ImageStore\image-20201127103524390.png" alt="image-20201127103524390" style="zoom:150%;" />





```python
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear

from vgg import VGG16BN  #load pretrained model

class FCN8s(fluid.dygraph.Layer):
    def __init__(self,num_classes=59)
    	super(FCN8s,self).__init__()
        backbone = VGG16BN(pretrained=False)
    	
        self.layer1 = backbone.layer1
        self.layer1[0].conv._padding=[100,100]
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer2 = backbone.layer2
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer3 = backbone.layer3
        self.pool3 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer4 = backbone.layer4
        self.pool4 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer5 = backbone.layer5
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        
        self.fc6 = Conv2D(512,4096,7, act = 'relu')
        self.fc7 = Conv2D(4096,4096,7, act = 'relu')
        self.drop6 = Dropout()
        self.drop7 = Dropout()
        
        self.score = Conv2D(4096,num_classes,1)
        self.score_pool3 = Conv2D(256,num_classes,1)
        
        self.score_pool4 = Conv2D(512,num_classes,1)
        
        self.up_output = Conv2DTranspose(num_channels=num_classes,
                num_filters=num_classes,
                filter_size=4,
                stride=2,
                bias_attr=False)
        self.up_output4 = Conv2DTranspose(num_channels=num_classes,
                num_filters=num_classes,
                filter_size=4,
                stride=2,
                bias_attr=False)
        self.up_final = Conv2DTranspose(num_channels=num_classes,
                num_filters=num_classes,
                filter_size=16,
                stride=8,
                bias_attr=False)
        
                
    def forwar(self, inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)   # 1/2
        x = self.layer2(inputs)
        x = self.pool2(x)	# 1/4
        x = self.layer3(inputs)
        x = self.pool3(x)	# 1/8
        pool3 = x
        x = self.layer4(inputs)
        x = self.pool4(x)	# 1/16
        pool4 = x
        x = self.layer5(inputs)
        x = self.pool5(x)	#
    	
        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)
    	
        x = self.score(x)
        x = self.up_output(x)
        
        up_output = x # 1/16
    	x = self.score_pool4(pool4)
        
        x = x[:, :, 5:5+up_output.shape[2],5:5+up_output.shape[3]]	
        up_pool4 = x
        x = up_pool4 + up_output
		x = self.up_pool4(x)
        up_pool4 = x
        
        x = self.score3(pool3)
        x = x[:, :, 9+9+up_pool4.shape[2], 9:9+up_pool4.shape[3]]
        up_pool3 = x #1/8
        
        x = up_pool3 + up_pool4
        
        x = self.u_final(x)
        
        x = x[:, :, 31:31+inputs.shape[2],31:31+inpits.shape[3]]
        
        
       
def main():
    with fluid.dygraph.guard(): #默认找GPU0
        x_data = np.random.rand(2,3,512,512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(num_classes =59 )
        model.eval()
        pred = model(x)
        print(pred.shape)
        
if __name__ == '__main__':
    main()
```



### basic trained

```python






class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale==1 else 1
    
    def __call__(self,image,label=None):
        imgae = image.astype(np.float32)
        image = image *self.val_scale
        image = image -self.mean
        image = image *(1/self.std)
        return image, label
    
class ConvertDataType(object):
    def __call__(self,image, label=None):
        if label is not None:
            label =label.astype(np.int64)
        return image.astype(np.float32), label
    
    
class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, label=None):
        image = cv2.resize(image,(self.size,self.size),interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        return image,label


def train(dataloader, model, criterion, optimizer, epoch, total_batch):
	model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
       	label = data[1]
        # 数据转置，方便同型
        image =fluid.layers.transpose(image,(0, 3, 1, 2))
        
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()
        
        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0],n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}],"+f"Step[{batch_id:04d}/{total_batch:04d}],"+f"Average Loss:{train_loss_meter.avg:4f}")
      return train_loss_meter.avg

```



### FCN-实践中的问题& 巧妙之处

 why Padding 100 

​	解决不同Input 尺寸问题，比如1x1，经过一个padding   (H+2xP-K)/s +1  得到一个尺寸，然后经过一个Pool转换为目标尺寸

 why cropping [5,9,31]

​	（下采样，Encoder）Pool层之后，得到各种尺寸的feature map， 在Up-Sampling操作时，是Conv_T操作，用上面公式的反公式逆推出新Feature map'的大小（可能存在个别Pixel差别），公式推出的上采样Feature map大小比下采样的Feature map小，所以在下采样过程中Crop部分图像

​	  ![image-20201128100954288](F:\Files\Typora_ImageStore\image-20201128100954288.png)

实际过程中用一个双线性插值就行了

 FCN优缺点

​	优点：

​	任意尺寸输入，效率高（GPU），FCN-8s，结合浅层信息

​	缺点：

​	分割结果不够精细，没有考虑上下文信息~=看看旁边有啥

![image-20201128101121414](F:\Files\Typora_ImageStore\image-20201128101121414.png)

## U-Net/PSP网络

### From FCN to U-Net  Architecture

![image-20201128104503403](F:\Files\Typora_ImageStore\image-20201128104503403.png)

实质是更像 Encoder-Decoder结构，其中有Concat操作

具体操作可换成线性模型，定义操作和求Feature map分开

![image-20201128105327174](F:\Files\Typora_ImageStore\image-20201128105327174.png)



![image-20201128110034660](F:\Files\Typora_ImageStore\image-20201128110034660.png)

Concat操作需要先去上层Crop同尺寸大小，然后堆叠起来

实际操作时可以去上采样过程中Padding 0，达到同样效果

![image-20201128110854271](F:\Files\Typora_ImageStore\image-20201128110854271.png)

使用1x1卷积在最后在channel得到了许多n_classes，使用Softmax得到一个概率分布，再取最大的概率响应，预测时使用argmax，把feature map变成1，而 w  h都不变，这就是分类结果

![image-20201128111208747](F:\Files\Typora_ImageStore\image-20201128111208747.png)

和FCN最大区别：和浅层信息融合，backbone不再是VGG



```python
import paddle
import paddle.fluid as fluid 
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2DTranspose

class Encoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
		#TODO: encoder contains:
        #		1 3x3conv + 1bn + relu +
        #		1 3x3conv + 1bn + relu
		# 		1 2x2 pool
        # return features before and after pool
        self.conv1 = Conv2D(num_channels,
                           num_filters,
                           filter_size=3,
                           stride=1,
                           padding=1)
        self.bn1 = BatchNorm(num_filters, act='relu')
        self.conv2 = Conv2D(num_filters,
     						num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn2 = BatchNorm(num_filters, act='relu')
		
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type='max', ceil_mode=True)
             
        
    def forward(self, inputs):
        # TODO: finish inference part
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_pooled = self.pool(x)
        # skip connectionx需要用，必须单独存
        
        return x, x_pooled
   
class Decoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Decoder,self).__init__()
        # TODO: decoder contains:
        #		1 2x2 transpose conv, stride=2, p=0(makes feature map 2x larger)
        #		1 3x3 conv + 1bn + 1relu+
		#		1 3x3 conv + 1bn + 1relu
        
        self.up = Conv2DTranspose(num_channels=num_channels,
                num_filters=num_filters,
                filter_size=2,
                stride=2)
        
        self.conv1 = Conv2D(num_channels,
                           num_filters,
                           filter_size=3,
                           stride=1,
                           padding=1)
        self.bn1 = BatchNorm(num_filters, act='relu')
        self.conv2 = Conv2D(num_filters,
     						num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn2 = BatchNorm(num_filters, act='relu')
		
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type='max', ceil_mode=True)
   
        
    def forward(self, inputs_prev, inputs):
		# TODO: forward contains on pad2d and Concat
        x = self.up(inputs)
        h_diff = (inputs_prev.shape[2] - x.shape[2])
        w_diff = (inputs_prev.shape[3] - x.shape[3])
        
        x = fluid.layers.pad2d(x, paddings=[h_diff/2, h_diff-h_diff//2, w_diff//2, w_diff- w_diff//2])
        x = fluid.layers.concat([inputs_prev,x],axis=1)
		x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Pad
        
		return x
    
class UNet(Layer):
    def __init__(self, num_classes=59):
        super(UNet, self).__init__()
        # encoder: 3->64->128->256->512
        # mid: 512->1024->1024
        
        #TODO: 4 encoders, 4 decoder ,and mid layers  contains 2x(1x1conv +bn + relu)
        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)
        
        self.mid_conv1 = Conv2D(521,1204, filter_size=1, padding = 0, stride =1)
        self.mid_bn1 = BatchNorm(1024,act='relu')
        self.mid_conv2 = Conv2D(1024,1024, filter_size=1, padding = 0, stride =1)
        self.mid_bn2 = BatchNorm(1024,act='relu')
        
        self.up4 = Decoder(1024,512)
        self.up3 = Decoder(512,256)
        self.up2 = Decoder(256,128)
        self.up1 = Decoder(128,64)
        # 把feature map变得和类别数一样
        self.last_conv = Conv2D(num_channels=64,num_filters=num_classes, filter_size=1)
        
    def forward(self, inputs):
        x1, x = self.down1(inputs)
        print(x1.shape,x.shape)
        x2, x = self.down2(x)
        print(x2.shape,x.shape)
        x3, x = self.down3(x)
        print(x3.shape,x.shape)
        x4, x = self.down4(x)
        print(x4.shape,x.shape)
           
        # middle layers
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)
        
        print(x4.shape, x.shape)
        x = self.up4(x4, x)
        print(x3.shape, x.shape)
        x = self.up3(x3, x)
        print(x2.shape, x.shape)
        x = self.up2(x2, x)
        print(x1.shape, x.shape)
        x = self.up1(x1, x)
         

def main():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        model = UNet(num_classes=59)
        x_data = np.random.rand(1, 3, 123, 123).astype(np.float32)
        inputs = to_variable(x_data)
        pred = model(inputs)
        
        print(pred.shape)
        
if __name__ == "__main__":
    main()
```



### PSP 分割网络

PSP Net : Pyramid Scene Parsing Network

Scene ~= Semantic Segmentation

FCN的缺点： 分割不够精细

​			**没有考虑上下文信息**

上下文信息（Context info):

​	·利用全局信息（global information)

​	·全局信息 in CNN ~= feture/pyramid

![image-20201130134846053](F:\Files\Typora_ImageStore\image-20201130134846053.png)

标记区域被识别成车，利用上下文信息（水）分割成船



PSP网络：

​	如何获取上下文信息： 增大感受野（Receptive Field)

什么是感受野（RF):

​	感受野 = 用于产生特征的输入图像中的区域大小

​	只针对于局部操作：例如 conv,pooling

![image-20201130135408741](F:\Files\Typora_ImageStore\image-20201130135408741.png)

![image-20201130135424038](F:\Files\Typora_ImageStore\image-20201130135424038.png)



![image-20201130140244705](F:\Files\Typora_ImageStore\image-20201130140244705.png)

模块是feature map,箭头是操作,第一步是adpative pooling(x,(,)) #输入和目标大小，平行操作。

第二步是降维 conv 1x1   /4

第三步是Up- Sampling

第五步是 Concat，把 up sampling 和原图C 融合

定义好操作

指向，谁连谁，在forward方法里面写

![image-20201130144545481](F:\Files\Typora_ImageStore\image-20201130144545481.png)

adaptive pool 是把h w 变化成size k

1x1 conv是h w不变，把channnel变化



adaptive pool可以任意输出大小，每次都要计算卷哪部分区域

![image-20201130145001647](F:\Files\Typora_ImageStore\image-20201130145001647.png)

计算严格按公式计算，在不规则区域取最大值

adaptive_pool_demo

```python
import paddle.fluid as fluid
import numpy as np
np.set_printtiotions(precision=2)

x = [
    []
]

def main():
    global x
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.arrary(x)
        print(x.shape)
        print(x)
        x = x[np.newaxis, np.newaxis, : ,: ]
        x = fluid.dygraph.to_variableA(x)
        y = fluid.layers.adaptive_pool2d(input=x,
                             pool_size=[5,3], #  pool_size == out_size
                             pool_type='max')
        y = y.numpy().squeeze((0,1))
        print(y.shape)
        print(y)
       
if __name__ == "__main__"
	main()
```



![image-20201130144033901](F:\Files\Typora_ImageStore\image-20201130144033901.png)

最前面是 bacnbone ,选取ResNet50/101 ，最后全连接使用1x1 conv，实现像素级分割

Coding

```python
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50

# pool with diofferent bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModulem, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters = num_channels // len(bin_size_list)
        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(
            	fluid.dygraph.Sequential(
                	Conv2D(num_channles, 					num_filters,1),
                    BatchNorm(num_filters, 						act='relu')
                )
            )
    def forward(self, inputs):
   		out = [inputs]
        # f 指代的是Sequnetial，把操作写成list
        for idx, f in enumerate(self.feature):
        	x = fluid.layers.adaptive_pool2d(inputs, self.bin_size_list[idx])
            x = f(x)
            x = fluid.layers.interpolate(x, inputs,shape[2::],align_corners=True)
            out.append(x)
    	# out  is LIST
        out = fluid.layers.concat(out, axis=1)   # NxCxHxW
    
    	return out
    
class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()
        
        res = ResNet50(pretrained=False)
        # stem: res.conv, res.pool2d_max
        self.layer0 = fluid.dygraph.Sequential(
        	res.conv,
        	res.pool2d_max
        )
        self.layer1 = res.layers1
        self.layer2 = res.layers2
        self.layer3 = res.layers3
        self.layer4 = res.layers4
		
        num_channels = 2048
        # psp: 2048 -> 2048*2
		self.pspmodule = PSPModule(num_channels, [1,2,3,6])
        num_channles *=2
        
        # cls: 2048*2 -> 512 -> num_classes
        self.classifier = fluid.dygraph.Sequential(
            Conv2D(num_channels=num_channels, num_filters=512, filter_size=3, padding=1),
 BatchNorm(512, act='relu'),
 Dropout(0.1),
 Conv2D(num_channels=512, num_filters=num_classes, filter_size=1)
        )
        # aux: 1024 -> 256 -> num_classes
        
    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(inputs)
        x = self.layer2(inputs)
        x = self.layer3(inputs)
        x = self.layer4(inputs)
		x = self.pspmodule(x)
        x = self.classifier(x)
        x = fluid.layers.interpolate(x, inputs.shape[2::],align_corners=True)
        
        return  x
        # aux: tmp_x = layer3
		
        
def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2,3,47,473).astype(np.float32)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        modlel.train()
        pred, aux = model(x)
        print(pred.shape, aux.shape)
        
if __main__ =="__main__"
	main()











```

PSP Net = Pyramid Scence Paring Network

简而言之：多尺度  具体操作： adaptive_average_pool

如何增加Backbone感受野？  Dilated Convolution

![image-20201130165332051](F:\Files\Typora_ImageStore\image-20201130165332051.png)

Dilated Conv = Atrous Conv = 空洞卷积

Dilated Conv ： 将Kernel 扩大填0

![image-20201130165703245](F:\Files\Typora_ImageStore\image-20201130165703245.png)



![image-20201130170140868](F:\Files\Typora_ImageStore\image-20201130170140868.png)

增大感受野，但没有增加计算量



## DeepLab 系列网络

From DeepLab V1 to V3

![image-20201130192516941](F:\Files\Typora_ImageStore\image-20201130192516941.png)

![image-20201130192753671](F:\Files\Typora_ImageStore\image-20201130192753671.png)

![image-20201130193128332](F:\Files\Typora_ImageStore\image-20201130193128332.png)

Classification 中 通过stride的调整来改变feature map大小，然后将各个classification 做相加

![image-20201130193350381](F:\Files\Typora_ImageStore\image-20201130193350381.png)

通过Dilation 增加感受野



DeepLab V2 网络结构

![image-20201130193438655](F:\Files\Typora_ImageStore\image-20201130193438655.png)

把feature map相加后没有上采样的操作，而是把label下采样，然后进行运算

![image-20201130194302469](F:\Files\Typora_ImageStore\image-20201130194302469.png)

中间黄色区域是不同的卷积操作，都是3x3，通过不同的dilation获取到不同的感受野，即上下文信息。

加padding 获取与原尺寸不变的feature map 64x64

与PSP网络不同之处： adaptive-pool 和 dilaton

![image-20201130195715720](F:\Files\Typora_ImageStore\image-20201130195715720.png)

先padding  然后用 dilation conv 得到与原来feature map一样的大小

dilated_conv.py

dilationn 越大，feature 越小，添加padding

```python
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer
import numpy as np
# np 打印精度
np.set_printiotions(precision=1)

def main():
    x = np.random.rand(6，6).astype(np.float32)
    print(x.shape)
    print(x)
    x = x[np.newaxis, np.newaxis, :, :]
    
    d = 2
    with fluid.dygraph.guard(fluid.CPUPlace()):
        w = np.ones(1,1, 3, 3).astype(np.float32)
        pa = fluid.ParaAttr(name='conv',
                           initializer=NumpyArrayInitializer(w))
        dilated_conv = Conv2D(num_channels=1,
   	   						  num_filters=1,
       						  filter_size=3,
                              dilation=d,
                              padding=2,
                              stride=1,
                              param_arrt=pa)
        print(dilated_conv.weight.numpy().squeeze((0,1)))
        print(f'dilation={d}')
        x = fluid.dygraph.to_variable(x)
    	y = y.numpy().squeeze((0,1))
        print(y.shape)
        print(y)
    
```



增大感受野之后，不需要全连接层

![image-20201130203504097](F:\Files\Typora_ImageStore\image-20201130203504097.png)

用了 Dialation 同时用 padding，现在很少用label下采样，普遍是对feature上采样

![image-20201130195827239](F:\Files\Typora_ImageStore\image-20201130195827239.png)

![image-20201130211413411](F:\Files\Typora_ImageStore\image-20201130211413411.png)

添加了1x1 Conv 目标是改变channel

添加了一个Adaptive pool + interpolation 把h w 变成1 ,就是一条线

Concat 和 Add 区别，add直接加，concat拼起来会把当前层的feature map变得特别大，下一层运算量暴增

![image-20201130212637741](F:\Files\Typora_ImageStore\image-20201130212637741.png)

前面层是相同的，后面层不同

![image-20201130212757275](F:\Files\Typora_ImageStore\image-20201130212757275.png)

![image-20201130212833956](F:\Files\Typora_ImageStore\image-20201130212833956.png)

前4层就说3 ，4，23，3 后面就是左图所示的堆叠块，每一份是一个ResBlock

![image-20201130213033334](F:\Files\Typora_ImageStore\image-20201130213033334.png)

下面层通过不同的dilation =2 4 8 改变感受野，进行实验（不知道哪个更好），论文说使用不同的dilation效果更好

![image-20201130214047816](F:\Files\Typora_ImageStore\image-20201130214047816.png)

伪代码



理论上模型可以任意输入，假设输入512x512,测试图大小不符

处理方法：1 Resize ， 一般是nearest，找附近最近像素填充，可能影响边界，但是效果还Ok

2.Diret

3.Sliding Window +padding

手写一个滑窗，类似一个conv，最后尺寸匹配不上+padding,在crop

DeepLabV3 coding

![image-20201202092216378](F:\Files\Typora_ImageStore\image-20201202092216378.png)

```python
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resmet_multi_grid import ResNet50, ResNet101, ResNet152


class ASPPPooling(Layer):
    def __init__(self, num_channels, num_filters):
        super(ASPPPooling, self).__init__()
        self.features =
        		fluid.dygraph.Sequential(
            	Conv2D(num_channels, num_filters,1 ),
            	BatchNorm(num_filters, act='relu')
                )
        
    def forward(self, inputs):
        n, c, h, w = inputs.shape
        
        x = fluid.layers.adaptive_pool2d(inputs,1)
        x = self.features(x)
        x = fluid.layers.interpolate(x, (h,w),align_corners=false)
        
        return x       


class ASPPConv(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_filters,dilation):
        super(ASPPConv, self).__init__(
        	Conv2D(num_channels, num_filters, filter_size=3,padding=dilation, dilation=dilation),
            BatchNorm(num_filters, act='relu')     
        	)

        
class ASPPModule(Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        self.features = []
        self.features.append(
        	fluid.dygraph.Sequential(
            	Conv2D(num_channels, num_filters,1 ),
            	BatchNorm(num_filters, act='relu')
        	    )
        	)
        self.features.append(ASPPPooling(num_channels, num_filters))
        for r in rates:
            self.features.append(
            	ASPPConv(num_channels, num_filters, r)
            	)     
        
        self.project =fluid.dygraph.Sequential(
        Conv2D(num_filters*(2 + len(rates)) , num_filters,1),
        BatchNorm(256, act='relu')
        )
    def forward(self, inputs):
        res = [] # 存结果
        for op in self.features:
            res.append(op(inputs))
        
        x = fluid.layers.concat(res, axis=1)
        x = self.project(x)
        
        return x 

    
class DeelLabHead(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_classes):
        super(DeepLabHead, self).__init__(
        	ASPPModule(num_channels, 256,[12, 24, 36]),
        	Conv2D(256,256, 3, padding=1),
            BatchNorm(256,act='relu'),
            Conv2D(256, num_classes,1)
        
        )
    
    
class Deeplab(Layer):
    def __init__(self, num_classes=59):
        super(Deeplab,self).__init__()
        resnet = ResNet50(pretrained=False)

		self.layer0 = fluid.dygraph.Sequential(
            resnet.conv,
            resnet.pool2d_max
        )
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
        # multigrid
		self.layer5 = resnet.layer5
		self.layer6 = resnet.layer6
		self.layer7 = resnet.layer7
        
        feature_dim = 2048
        self.classifier = DeepLabHead(feature_dim,num_classes)
     
    def forward(self, inputs):
        n, c, h, w = inputs.shape
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.classifier(x)
        x = fluid.layers.interpolate(x, (h,w),align_corners=false)
        
        return x


def main()
	with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variaable(x_data)
        model = DeepLab(num_classes=59)
        model.eval()
		# pred, score_map = model(x)
        pred = model(x)
        # print(pred.shape, score_map.shape)
        print(pred.shape)
        
if __name__ == '__main__':
    main()



```

代码无需一行一行看，主要先看forward方法

可以换backbone,然后找每层干嘛了



## 图卷积神经网络

![image-20201202095345986](F:\Files\Typora_ImageStore\image-20201202095345986.png)

消息传递网络，W可学习

![image-20201202095443203](F:\Files\Typora_ImageStore\image-20201202095443203.png)

只考虑语义分割的上下文

![image-20201202101240395](F:\Files\Typora_ImageStore\image-20201202101240395.png)

![image-20201202194141381](F:\Files\Typora_ImageStore\image-20201202194141381.png)

类似于，把相似的像素聚类

![image-20201202194341486](F:\Files\Typora_ImageStore\image-20201202194341486.png)

![image-20201202194951202](F:\Files\Typora_ImageStore\image-20201202194951202.png)

下面不同的方法：利用语义

![image-20201202200224634](F:\Files\Typora_ImageStore\image-20201202200224634.png)

![image-20201202200353838](F:\Files\Typora_ImageStore\image-20201202200353838.png)

![image-20201202205442127](F:\Files\Typora_ImageStore\image-20201202205442127.png)



## 图像的实例分割与全景分割入门

1 基本概念

![image-20201203103517597](C:\Users\Han\OneDrive\桌面\image-20201203103517597.png)

object 可以考虑个数

stuff 比如天空和陆地，没有明显边界，无法用数量区分

instance seg要区分出pixel 如具体是车子，并且具体属于哪个车

mask 得到一个框，框内

2.经典方法

实例分割：

​	Porposal-based/Top-down 候选区域选取-物体

​		Mask RCNN ICCV2017

​	Porposal-free/Bottom-up 一步完成

​		SOLO ECCV2020

### Mask-RCNN

![image-20201203104410366](F:\Files\Typora_ImageStore\image-20201203104410366.png)

首先给图像一个bounding box框，然后给出框内每一个 像素属于哪类

RPN 扣出一个候选框（哪些是物体）， 然后提取特征图（包含图像信息 ），再做分类回归，预测，生成mask

![image-20201203104838794](F:\Files\Typora_ImageStore\image-20201203104838794.png)

RPN 利用滑窗在feature map 考虑每个pixel属于正负样本的可能性，利用不同的anchor,画出bounding box，再利用resnet101提取特征，得到不同分辨率，然后predict

特征金字塔通过输入单一的feature map,来自backbone，通过不同的pooling和padding获得不同尺度

mask对应的ground truth是怎么生成的

![image-20201203155540109](F:\Files\Typora_ImageStore\image-20201203155540109.png)

一张图像中的两个实物有不同的中心位置或者尺寸 （位置和大小）

![image-20201203160852895](F:\Files\Typora_ImageStore\image-20201203160852895.png)

这个跟Mask-RCNN 的一个区别就是不用预测Box

![image-20201203161625903](F:\Files\Typora_ImageStore\image-20201203161625903.png)

![image-20201203162242098](F:\Files\Typora_ImageStore\image-20201203162242098.png)

一个Semantic 一个Instance head 类似于Deeplab 的ASPP PSPModule，再融合

![image-20201203162427504](F:\Files\Typora_ImageStore\image-20201203162427504.png)

在得到语义分割头和实例分割头之后再作为输入，进行全景分割

为实例分割添加一个通道，来弥补预测中漏选的实例（一般不可能全部预测出来，实例个数+种类）

![image-20201203164100832](F:\Files\Typora_ImageStore\image-20201203164100832.png)

实例分割中已经预测出了类别，直接输出送到全景分割。全景分割后的结果是一个概率。

没有预测成功的object类别可用原类别（最大类别）-已预测的类别

即总类别的概率 H x W x 类别（通道数，以概率方式体现）- 每个像素值属于某个类别的概率，只 保留最大的， 体现形式是把feature map压扁，从三维变成二维，然后做减法，得到漏选的实例，放到最后补充的通道

![image-20201203165409946](F:\Files\Typora_ImageStore\image-20201203165409946.png)

Panoptic-DeepLab

![image-20201203165548697](F:\Files\Typora_ImageStore\image-20201203165548697.png)

![image-20201203165754484](F:\Files\Typora_ImageStore\image-20201203165754484.png)

在backbone 使用了一个膨胀卷积，使feature没有直接变小到1/32

用了两种ASPP

![image-20201203185905325](F:\Files\Typora_ImageStore\image-20201203185905325.png)

实例分割类别如何确定？

​	实际是由语义分割结果确定的，概率

Bottom up 结构相对更简单

## 复习

​	train 方法，dropout和evaluate 的behavor是不同的

dropout 训练是取概率的，预测时候去掉

​	

​	__call__  类的实例或者说类创建的对象直接当函数用



​	forward  raise 是 错误处理，一般没有实现，需要自己写



​	Sequential 把所有layer做了一个前向



​	init 和 call方法 ，init初始化类的实例，下次需要调用用call方法

​	

​	OpenCV 、 Interpolation

![image-20201203192200905](F:\Files\Typora_ImageStore\image-20201203192200905.png)

​	Numpy  transpose 

​	需要清楚每一步处理的矩阵(data,label)到底是一个什么格式，维度

![image-20201203192403158](F:\Files\Typora_ImageStore\image-20201203192403158.png)

3D-Unet 三维医学图像



![image-20201203192950110](F:\Files\Typora_ImageStore\image-20201203192950110.png)

![image-20201203193007453](F:\Files\Typora_ImageStore\image-20201203193007453.png)

读batch 读到tensor

![image-20201203193405387](F:\Files\Typora_ImageStore\image-20201203193405387.png)

???一般是变大![image-20201203193505078](F:\Files\Typora_ImageStore\image-20201203193505078.png)





![image-20201203193749331](F:\Files\Typora_ImageStore\image-20201203193749331.png)



案例

![image-20201203194906056](F:\Files\Typora_ImageStore\image-20201203194906056.png)

![image-20201203195005969](F:\Files\Typora_ImageStore\image-20201203195005969.png)



![image-20201203195025714](F:\Files\Typora_ImageStore\image-20201203195025714.png)





![image-20201203195355890](F:\Files\Typora_ImageStore\image-20201203195355890.png)

![image-20201203195408105](F:\Files\Typora_ImageStore\image-20201203195408105.png)

Deeplav v3  多尺度缩放



![image-20201203195604011](F:\Files\Typora_ImageStore\image-20201203195604011.png)



![image-20201203200120663](F:\Files\Typora_ImageStore\image-20201203200120663.png)