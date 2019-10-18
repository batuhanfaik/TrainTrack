import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Lambda
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")


class TrainDataReaderWithHSL(torch.utils.data.Dataset):
    def __init__(self, mode, directory, categories):
        assert isinstance(directory, str), "Provide a directory as string"
        assert isinstance(categories, list), "Provide categories as list of classes training set directories"
        super(TrainDataReaderWithHSL, self).__init__()

        self.directory = directory
        self.categories = categories

        self.train_img_paths = []
        self.train_target_classes = []
        self.test_img_paths = []
        self.test_target_classes = []
        self.mode = mode

        # Full ImageNet values
        # self.rgb_mean_std = {"mean": (0.485, 0.456, 0.406),
        #                      "std": (0.229, 0.224, 0.225)}
        # Only market dataset values
        self.rgb_mean_std = {"mean": (0.5074962800396952, 0.5093141510901613, 0.509899199283156),
                             "std": (0.33326811293209835, 0.3329321276571116, 0.3320949847327579)}

        self.rgb_transform = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop((224, 224)), transforms.ToTensor(),
             transforms.Normalize(mean=self.rgb_mean_std["mean"], std=self.rgb_mean_std["std"])])

        # self.rgb_transform = transforms.Compose(
        #     [transforms.Resize(256),
        #      transforms.FiveCrop((224, 224)),
        #      Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))])

        for i, categ in enumerate(self.categories):
            train_imgs = os.listdir(self.directory + categ + "/train/")
            test_imgs = os.listdir(self.directory + categ + "/test/")
            train_imgs = sorted(train_imgs)
            test_imgs = sorted(test_imgs)

            for _, img in enumerate(train_imgs):
                self.train_img_paths.append(self.directory + categ + "/train/" + img)
                self.train_target_classes.append(i)

            for _, img in enumerate(test_imgs):
                self.test_img_paths.append(self.directory + categ + "/test/" + img)
                self.test_target_classes.append(i)

        self.train_image_count = len(self.train_target_classes)
        self.test_image_count = len(self.test_target_classes)

        if mode == "train":
            self.input_img_paths = self.train_img_paths
            self.target_classes = self.train_target_classes
            self.image_count = self.train_image_count

        if mode == "test":
            self.input_img_paths = self.test_img_paths
            self.target_classes = self.test_target_classes
            self.image_count = self.test_image_count

        print(mode + " set size is", self.image_count)

    def load_input_img(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.target_classes)

    def __getitem__(self, index):
        input_img = self.load_input_img(self.input_img_paths[index])
        normalized = False
        rgb_img = input_img
        return self.rgb_transform(rgb_img), self.target_classes[index]


class TestDataReaderWithHSL(torch.utils.data.Dataset):

    def __init__(self, mode, directory, categories):
        assert isinstance(directory, str), "Provide a directory as string"
        assert isinstance(categories, list), "Provide categories as list of classes test set directories"

        super(TestDataReaderWithHSL, self).__init__()

        self.directory = directory
        self.categories = categories

        self.train_img_paths = []
        self.train_target_classes = []
        self.test_img_paths = []
        self.test_target_classes = []
        self.mode = mode

        # Full ImageNet values
        # self.rgb_mean_std = {"mean": (0.485, 0.456, 0.406),
        #                      "std": (0.229, 0.224, 0.225)}
        # Only market dataset values
        self.rgb_mean_std = {"mean": (0.5074962800396952, 0.5093141510901613, 0.509899199283156),
                             "std": (0.33326811293209835, 0.3329321276571116, 0.3320949847327579)}

        # self.rgb_transform = transforms.Compose(
        #     [transforms.Resize(224), transforms.CenterCrop((224, 224)), transforms.ToTensor(),
        #      transforms.Normalize(mean=self.rgb_mean_std["mean"], std=self.rgb_mean_std["std"])])

        self.rgb_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.FiveCrop((224, 224)),
             Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))])

        for i, categ in enumerate(self.categories):
            train_imgs = os.listdir(self.directory + categ + "/train/")
            test_imgs = os.listdir(self.directory + categ + "/test/")
            train_imgs = sorted(train_imgs)
            test_imgs = sorted(test_imgs)

            for _, img in enumerate(train_imgs):
                self.train_img_paths.append(self.directory + categ + "/train/" + img)
                self.train_target_classes.append(i)

            for _, img in enumerate(test_imgs):
                self.test_img_paths.append(self.directory + categ + "/test/" + img)
                self.test_target_classes.append(i)

        self.train_image_count = len(self.train_target_classes)
        self.test_image_count = len(self.test_target_classes)

        if mode == "train":
            self.input_img_paths = self.train_img_paths
            self.target_classes = self.train_target_classes
            self.image_count = self.train_image_count

        if mode == "test":
            self.input_img_paths = self.test_img_paths
            self.target_classes = self.test_target_classes
            self.image_count = self.test_image_count

        print(mode + " set size is", self.image_count)

    def load_input_img(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.target_classes)

    def __getitem__(self, index):
        input_img = self.load_input_img(self.input_img_paths[index])
        normalized = False
        rgb_img = input_img
        return self.rgb_transform(rgb_img), self.target_classes[index]
