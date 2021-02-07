import os.path as osp
import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2
from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals(object):
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations(object):
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

@PIPELINES.register_module()
class LoadPasetImages(LoadImageFromFile):
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, class_names=None,
                       base_cls_num=0,
                       image_root='',
                       prob=0.5,
                       max_paste_num=3,
                       **kwargs):
        # TODO: add more filter options
        assert isinstance(class_names, list)
        self.class_names = class_names
        self.catID = {}
        for class_name in self.class_names:
            self.catID[class_name] = base_cls_num+1
        self.image_root = image_root
        self.prob = prob
        self.max_paste_num = max_paste_num
        super(LoadPasetImages, self).__init__(**kwargs)
        self.pipline = self.Compose([self.RandomRotate(),
                                     self.RandomRadiusBlur(),
                                     self.HomoGrapy(),
                                     self.RandomlFlip(),
                                     self.RandomErase()])
        self.ICON_FACTOR = 0.3
    class HomoGrapy(object):
        def __init__(self, prob=0.5, noise=100):
            self.noise = noise
            self.prob = prob

        def __call__(self, input):
            image = input
            image_width = image.shape[1]
            image_height = image.shape[0]
            prob = np.random.random()

            if prob > self.prob:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                index_x, index_y = np.where(gray_image == 0)

                x1 = np.min(index_x)
                x2 = np.max(index_x)
                y1 = np.min(index_y)
                y2 = np.max(index_y)

                while True:
                    noise_x1 = np.random.randint(-self.noise, self.noise)
                    noise_x2 = np.random.randint(-self.noise, self.noise)
                    noise_y1 = np.random.randint(-self.noise, self.noise)
                    noise_y2 = np.random.randint(-self.noise, self.noise)
                    new_x1 = max(0, x1 + noise_x1)
                    new_x2 = min(image_width, x2 + noise_x2)
                    new_y1 = max(0, y1 + noise_y1)
                    new_y2 = min(image_height, y2 + noise_y2)

                    if new_x2 > new_x1 and new_y2 > new_y1:
                        break

                src_points = np.array([np.array([x1, y1]), np.array([x1, y2]), np.array([x2, y1]), np.array([x2, y2])])
                dst_points = np.array(
                    [np.array([new_x1, new_y1]), np.array([new_x1, new_y2]), np.array([new_x2, new_y1]),
                     np.array([new_x2, new_y2])])
                trans_matrix, _ = cv2.findHomography(src_points, dst_points)

                image = cv2.warpPerspective(image, trans_matrix, (image_width, image_height))

            return image

    class RandomlFlip(object):
        def __init__(self, prob=0.5):
            self.prob = prob

        def __call__(self, input):
            if np.random.random() < self.prob:
                image = input[::-1, ::-1, :]
            else:
                image = input
            return image

    class RandomErase(object):
        def __init__(self, prob=0.5, dsize=0.3):
            self.prob = prob
            self.dsize = dsize

        def __call__(self, input):
            image = input
            image_width = image.shape[1]
            image_height = image.shape[0]
            prob = np.random.random()

            if prob > self.prob:
                x_erase_size = int(np.random.random() * self.dsize * image_width)
                y_erase_size = int(np.random.random() * self.dsize * image_height)

                x1 = np.random.randint(0, image_width - x_erase_size)
                y1 = np.random.randint(0, image_height - y_erase_size)
                image[y1:y1 + y_erase_size, x1:x1 + x_erase_size, :] = 0

            return image

    class Compose(object):
        def __init__(self, transforms):
            assert isinstance(transforms, list)
            self.transform_list = transforms

        def __call__(self, input):
            image = input
            for transform in self.transform_list:
                image = transform(image)
            return image

    class RandomRotate(object):
        def __init__(self, rotate=30, prob=0.5):
            self.angle = 30
            self.prob = 0.5

        def __call__(self, input):
            prob = np.random.random()
            image = np.array(input)
            angle = np.random.randint(-self.angle, self.angle)
            if prob > self.prob:
                SrcWidth, SrcHeight = image.shape[1], image.shape[0]
                DstWidth = image.shape[0] * np.sin(abs(angle / 180 * np.pi)) + image.shape[1] * np.cos(
                    abs(angle / 180 * np.pi))
                DstHeight = image.shape[0] * np.cos(abs(angle / 180 * np.pi)) + image.shape[1] * np.sin(
                    abs(angle / 180 * np.pi))

                top, bottom, left, right = [max(0, int(i)) for i in
                                            [(DstHeight - SrcHeight) // 2,
                                             DstHeight - SrcHeight - (DstHeight - SrcHeight) // 2,
                                             (DstWidth - SrcWidth) // 2,
                                             DstWidth - SrcWidth - (DstWidth - SrcWidth) // 2]]

                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                RotateMatrix = cv2.getRotationMatrix2D((DstWidth / 2, DstHeight / 2), angle, 1)
                image = cv2.warpAffine(image, RotateMatrix, (int(DstWidth), int(DstHeight)))

            return image

    class RandomRadiusBlur(object):
        def __init__(self, radius=5, prob=0.5, std=0):
            self.prob = prob
            self.radius = radius
            self.std = std

        def __call__(self, image):
            CurProb = np.random.random()
            if CurProb > self.prob:
                image = np.array(image)
                radius = np.random.randint(1, self.radius) // 2 * 2 + 1
                image = cv2.GaussianBlur(image, (radius, radius), self.std)
            return image

    def Embedding(self, image, icon):
        gray_icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        index_x, index_y = np.where(gray_icon != 0)

        x1 = np.min(index_x)
        x2 = np.max(index_x)
        y1 = np.min(index_y)
        y2 = np.max(index_y)

        icon = icon[x1:x2, y1:y2, :]

        image_height, image_width = image.shape[:2]
        icon_height, icon_width = icon.shape[:2]

        max_size = (np.random.random() + 1) / 2 * self.ICON_FACTOR * min(image_width, image_height)
        max_icon_size = max(icon_height, icon_width)
        factor = max_size / max_icon_size

        icon_height, icon_width = int(icon_height * factor), int(icon_width * factor)

        icon = cv2.resize(icon, (icon_width, icon_height))

        icon_gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

        x1 = np.random.randint(0, image_width - icon_width)
        y1 = np.random.randint(0, image_height - icon_height)

        roi = image[y1:y1 + icon_height, x1:x1 + icon_width, :]
        mask = icon_gray != 0

        roi[mask, :] = icon[mask, :]
        image[y1:y1 + icon_height, x1:x1 + icon_width, :] = roi

        return np.array([np.float32(x1), np.float32(y1), np.float32(icon_width+x1), np.float32(icon_height+y1)])

    def IouCal(self, Box1, Box2):
        inner_x1 = max(Box1[0], Box2[0])
        inner_y1 = max(Box1[1], Box2[1])
        inner_x2 = min(Box1[2], Box2[2])
        inner_y2 = min(Box1[3], Box2[3])

        if inner_x2 - inner_x1 < 0 or inner_y2 - inner_y1 < 0:
            return 0
        area_inner = (inner_x2 - inner_x1) * (inner_y2 - inner_y1)
        area = (Box2[2] - Box2[0]) * (Box2[3] - Box2[1]) + \
               (Box1[2] - Box1[0]) * (Box1[3] - Box1[1]) - \
               area_inner
        return max(0, area_inner / area)

    def __call__(self, results):
        for class_name in self.class_names:
            image_root = os.path.join(self.image_root, class_name)
            assert osp.isdir(image_root)
            image_paths = os.listdir(image_root)
            num_image = len(image_paths)
            paste_num = np.random.randint(0, self.max_paste_num) #随机粘贴几张图片
            for _ in range(paste_num):
                flag = True
                if self.file_client is None:
                    self.file_client = mmcv.FileClient(**self.file_client_args)

                paste_index = np.random.randint(0, num_image)
                image_name = image_paths[paste_index]
                filename = osp.join(image_root, image_name)

                img_bytes = self.file_client.get(filename)
                paste_image = mmcv.imfrombytes(img_bytes, flag=self.color_type)
                if self.to_float32:
                    paste_image = paste_image.astype(np.float32)
                    paste_image = self.pipline(paste_image)

                img = results['img']
                new_bbox = self.Embedding(img, paste_image)
                for bbox in results['gt_bboxes']:
                    iou = self.IouCal(new_bbox, bbox)
                    if iou > 0.3:
                        flag = False  # 如果重叠度过高，不叠加该mask
                        break
                if not flag:
                    continue

                results['gt_bboxes'] = np.append(results['gt_bboxes'], new_bbox).reshape(-1, 4)
                results['gt_labels'] = np.append(results['gt_labels'], self.catID[class_name])


                # bboxes = results['gt_bboxes']
                # for box in bboxes:
                #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
                # cv2.imwrite('../test_image/{}'.format(image_name), img)

        return results


        # assert 'gt_bboxes' in results
        # gt_bboxes = results['gt_bboxes']
        # w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        # h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        # keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        # if not keep.any():
        #     return None
        # else:
        #     keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
        #     for key in keys:
        #         if key in results:
        #             results[key] = results[key][keep]
        #     return results