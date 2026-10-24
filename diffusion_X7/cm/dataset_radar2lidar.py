import os
import cv2
import torch
from torch.utils.data import Dataset


class Radar2LidarDataset(Dataset):
    """
    返回格式必须是:
        cond, batch, name

    cond  -> Radar  (condition, y)
    batch -> LiDAR  (diffusion target x)
    name  -> 文件名 (可用于 debug)
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        num_past_frames=0,
        start_index=0,
        end_index=None,
        cache_images=False,
    ):
        """
        root_dir/
          ├── radar/  (128x128 grayscale png)
          └── lidar/  (128x128 grayscale png)
        """
        self.radar_dir = os.path.join(root_dir, "radar")
        self.lidar_dir = os.path.join(root_dir, "lidar")

        radar_files = sorted(
            name
            for name in os.listdir(self.radar_dir)
            if os.path.isfile(os.path.join(self.radar_dir, name))
        )
        lidar_files = sorted(
            name
            for name in os.listdir(self.lidar_dir)
            if os.path.isfile(os.path.join(self.lidar_dir, name))
        )

        assert len(radar_files) == len(lidar_files), \
            "Radar / LiDAR 数量不一致"
        assert radar_files == lidar_files, "Radar / LiDAR 文件名不一致"

        if end_index is None:
            end_index = len(radar_files)
        if not 0 <= start_index < end_index <= len(radar_files):
            raise ValueError(
                f"invalid range [{start_index}, {end_index}) for {root_dir} "
                f"with {len(radar_files)} frames"
            )
        self.start_index = start_index
        self.end_index = end_index
        self.radar_files = radar_files[start_index:end_index]
        self.lidar_files = lidar_files[start_index:end_index]

        self.transform = transform
        self.num_past_frames = num_past_frames
        self.image_cache = {}
        if cache_images:
            for directory, filenames in (
                (self.radar_dir, self.radar_files),
                (self.lidar_dir, self.lidar_files),
            ):
                for filename in filenames:
                    image_path = os.path.join(directory, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    assert image is not None, f"读取失败: {image_path}"
                    self.image_cache[image_path] = image

    def __len__(self):
        return len(self.radar_files)

    def _load_image(self, image_path):
        image = self.image_cache.get(image_path)
        if image is None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            assert image is not None, f"读取失败: {image_path}"

        if self.transform is not None:
            return self.transform(image)
        return torch.from_numpy(image).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx):
        lidar_path = os.path.join(self.lidar_dir, self.lidar_files[idx])
        lidar = self._load_image(lidar_path)

        radar_frames = []
        for offset in range(self.num_past_frames, -1, -1):
            hist_idx = max(0, idx - offset)
            radar_path = os.path.join(self.radar_dir, self.radar_files[hist_idx])
            radar_frames.append(self._load_image(radar_path))

        radar = torch.cat(radar_frames, dim=0)
        name = self.radar_files[idx]

        # ⚠️ 顺序非常重要：cond 在前，batch 在后
        return radar, lidar, name
