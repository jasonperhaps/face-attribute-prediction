import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
import os
import os.path

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CelebA(data.Dataset):
    def __init__(self, img_dir=None, split='train', transform=None, label_transform=None, loader=default_loader):
        super().__init__()
        input_dict = None
        import pickle
        if split == 'train':
            with open('./split/train.pickle', 'rb') as handle:
                input_dict = pickle.load(handle)
        elif split == 'val':
            with open('./split/val.pickle', 'rb') as handle:
                input_dict = pickle.load(handle)
        else:
            with open('./split/test.pickle', 'rb') as handle:
                input_dict = pickle.load(handle)
            
        self.imgs = [os.path.join(img_dir,file) for file in input_dict['imgs']]
        self.labels = input_dict['labels'] 
        self.transform = transform 
        self.label_transform = label_transform
        self.loader = loader 

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index],dtype=np.int64))
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))
        if self.label_transform is not None:
            try:
                label = self.label_transform(label)
            except:
                pass
        return img,label

