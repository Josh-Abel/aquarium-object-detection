import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors

class train_augmentations:

    def __init__(self):

        transforms = []
        
        transforms += [T.Resize(size=(224, 224), antialias=True)]    
        transforms += [T.RandomChoice([T.RandomHorizontalFlip(), T.RandomAffine(degrees=(-90, 90)), T.RandomVerticalFlip()]),
                        T.RandomChoice([T.RandomChannelPermutation(), T.RandomGrayscale(), T.RandomPhotometricDistort(), T.GaussianBlur(kernel_size=3)])]
        

        transforms += [T.ToImage()]

        transforms += [T.ToDtype(torch.float32, scale=True)]

        
        transforms += [
            T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
            T.SanitizeBoundingBoxes(),
            T.ToPureTensor(),
        ]
        
        transforms += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class val_augmentations:
    def __init__(self):
        transforms = []
        
        
        transforms += [T.Resize(size=(224, 224), antialias=True)]    
        transforms += [T.ToImage()]
        

        transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
