"""
Custom dataset class for loading PNG format images and masks
"""
from os.path import join
from imageio import imread
from skimage import measure
import numpy as np


class MyDataset:
    """Custom dataset class"""
    
    def __init__(self, directory, metadata_file, **label_kwargs):
        """
        Args:
            directory: Root directory of dataset (contains img/ and mask/ folders)
            metadata_file: Path to metadata file (e.g., 'metadata/training.txt')
            **label_kwargs: Parameters passed to skimage.measure.label()
        """
        self.img_dir = join(directory, 'img')
        self.mask_dir = join(directory, 'mask')
        self.metadata_file = metadata_file
        
        # Read file list
        self.names = self._read_metadata(metadata_file)
        if not self.names:
            raise ValueError(f"Unable to read {metadata_file}")
        
        # Load all images
        self.images, self.masks, self.labels = self._load_data(**label_kwargs)
    
    def _read_metadata(self, filepath):
        """Read metadata file"""
        try:
            with open(filepath, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
    
    def _load_data(self, **label_kwargs):
        """Load all images and masks"""
        images = []
        masks = []
        labels = []
        
        for fname in self.names:
            # Read image
            img_path = join(self.img_dir, fname)
            img = imread(img_path)
            images.append(img)
            
            # Read mask
            mask_path = join(self.mask_dir, fname)
            mask = imread(mask_path)
            masks.append(mask)
            
            # Generate instance labels
            if len(mask.shape) == 3:
                mask_2d = mask[:, :, 0]
            else:
                mask_2d = mask
            label = measure.label(mask_2d, **label_kwargs)
            labels.append(label)
        
        return images, masks, labels
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, item):
        """Return (filename, image, mask, label)"""
        return self.names[item], self.images[item], self.masks[item], self.labels[item]


class MyDatasetTrain(MyDataset):
    """Training set"""
    def __init__(self, directory, **label_kwargs):
        super().__init__(directory, join(directory, 'metadata', 'training.txt'), **label_kwargs)


class MyDatasetVal(MyDataset):
    """Validation set"""
    def __init__(self, directory, **label_kwargs):
        super().__init__(directory, join(directory, 'metadata', 'validation.txt'), **label_kwargs)


class MyDatasetTest(MyDataset):
    """Test set"""
    def __init__(self, directory, **label_kwargs):
        super().__init__(directory, join(directory, 'metadata', 'test.txt'), **label_kwargs)
