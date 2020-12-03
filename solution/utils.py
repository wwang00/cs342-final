from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, 'player*_img.png')):
            self.files.append(im_f.replace('_img.png', ''))
        self.transform = transform
        self.min_size = min_size

    def _filter(self, boxes):
        if len(boxes) == 0:
            return boxes
        return boxes[abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0]) >= self.min_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        im = Image.open(b + '_img.png')
        depth = Image.open(b + '_depth.png')
        nfo = np.load(b + '.npz')
        data = im, depth, self._filter(nfo['kart']), self._filter(nfo['projectile']), self._filter(nfo['nitro']), self._filter(nfo['pickup'])
        if self.transform is not None:
            data = self.transform(*data)
        return data


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('box9')
    import code
    code.interact(local=locals())
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, im2, kart, bomb, pickup, _ = dataset[i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('dense_data/train',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):

        im, *dets = dataset[i]
        hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[1:])
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
