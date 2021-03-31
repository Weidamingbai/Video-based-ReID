import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import random
from data_process import dataset_manager
from IPython import embed
import torchvision.transforms as T


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img_paths, pid, camid = self.dataset[index]


        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            # print("取样方式维random")
            # range(0, 16)
            frame_indices = range(num)
            # rand_end =11
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            # range(6, 10)
            indices = frame_indices[begin_index:end_index]
            indices=np.array(indices)
            # 这个地方要实现的逻辑是 当我的num不够长的时候 追加
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                # 正常不会运行到 运行到出错
                print("adhbakjvgkgbvaksdjhbgvkladsbvglkabsvlha")
                indices = np.append(indices,index)

            imgs = []
            for index in indices:
                # 转成int类型
                index=int(index)
                # 拿出对应的单张图片
                img_path = img_paths[index]
                # 读图片
                img = read_image(img_path)
                # 进行数据增强及转换
                if self.transform is not None:
                    img = self.transform(img)
                # img.Size([3, 224, 112])
                #在最外层增加一个维度 用来存放batchsize
                img = img.unsqueeze(0)
                imgs.append(img)
            # 此时imgs是一个列表 len(imgs)=15 对应第一个的长度 存放了15个tensor [1,3,224,112] [b,c,h,w]
            # cat 将两个张量连接起来
            imgs = torch.cat(imgs, dim=0)
            # imgs.Size([15, 3, 224, 112])
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            # num =16?
            # 返回的是一个可迭代对象 对应一个身份的图片个数
            # frame_indices
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                # 里边放了一个range
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            # 这里应该是要把num的最后一部分放在这里吧   剩下的一部分不够一个seq_len
            last_seq=frame_indices[cur_index:]
            last_seq=np.array(last_seq)
            index = len(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq = np.append(last_seq,index)
                while(len(last_seq)!=4):
                    last_seq = np.append(last_seq,index)

            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    #img = T.ToTensor(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    # list index out of range错误出现的原因主要有两个，一个可能是下标超出范围，一个可能是list是空的，没有一个元素，
                    # print(imgs[index].size())

                # imgs =[16,3,224,112] [s,c,h,w]
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            # torch.Size([1, 16, 3, 224, 112])  [b,s,c,h,w]
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

# 验证
if __name__ == "__main__":
    dataset =dataset_manager.init_dataset(name="mars")
    transform_test = T.Compose([
        T.Resize((224, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = VideoDataset(dataset.query,sample='dense',seq_len=4,transform=transform_test)
    for batch_id,(img,pid,camid) in enumerate(train_loader):
        break
    print(batch_id,img,pid,camid)
