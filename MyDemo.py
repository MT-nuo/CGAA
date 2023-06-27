import cv2
import torch
import utils.config as config
from model import build_segmenter
from utils.dataset import tokenize
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as T


cfg = config.load_cfg_from_cfg_file("./config/refcoco/cris_r101.yaml")
PATH = "exp/refcoco/RN101_32_60e_allFusionFs_CRIS_Phrase_5_Area_200_layer4_ks3/best_model.pth"
# {CRIS_Phrase_5_Area_200_layer4, RN101_32_60e_allFusionFs_CRIS_Phrase_5_Area_200_layer4_ks3
# CRIS_Area_200, RN50_32_60e_allFusionFs_CRIS_Phrase_5_Area_200_layer4_ks3}
model, _ = build_segmenter(cfg)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
print('model:', next(model.parameters()).device)  # 输出：cuda:0
print("=> loaded checkpoint '{}'".format(PATH))

input_size = (416, 416)


def getTransformMat(img_size, inverse=False):
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None

def convert(img):
    img_size = img.shape[:2]
    mat, mat_inv = getTransformMat(img_size, False)
    img = cv2.warpAffine(
        img,
        mat,
        input_size,
        flags=cv2.INTER_CUBIC,
        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
    pad_img = img
    
    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()

    mean = torch.tensor([0.48145466, 0.4578275,
                         0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                        0.27577711]).reshape(3, 1, 1)
    img.div_(255.).sub_(mean).div_(std)
    return img, pad_img

def overlay_davis(image, mask, colors=[[0, 0, 0], [100, 255, 255]], cscale=1, alpha=0.4):  
    # [244, 164, 96], [0, 255, 255], [160, 223, 223]
    from scipy.ndimage import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)

if __name__ == '__main__':
    img = cv2.imread("./Vis/COCO_train2014_000000580741.jpg")
    # COCO_train2014_000000010471; COCO_train2014_000000027950; COCO_train2014_000000003478
    # COCO_train2014_000000189864(zebra in the middle); COCO_train2014_000000188831(white cat)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resized_img = cv2.resize(img, (416, 416), interpolation= cv2.INTER_LINEAR)
    # img_ndarray = np.array(img)
    # original_h, original_w = img_ndarray.shape[0], img_ndarray.shape[1]
    img, pad_img = convert(img)
    img = img.unsqueeze(0)
    # print('img_ndarray shape: ', img_ndarray.shape)
    
    sent = "brown sofa at right where two goofy kids are sitting"   # small elephant; guy in hat; person carrying black bag; cow behind the tree
    text = tokenize(sent, 17, True)
    # text = text.cuda(non_blocking=True)
    pred = model(img, text)

    pred = torch.sigmoid(pred)
    # pred = F.interpolate(pred.float(), (original_h, original_w))
    pred = F.interpolate(pred.float(), (416, 416))
    pred = pred.squeeze()
    print('pred shape: ', pred.shape)
    pred = pred.cpu().detach().numpy()
    pred = np.array(pred > 0.35)
    # pred = np.array(pred * 255, dtype=np.uint8)
    # plt.imshow(pred[0, 0])
    # plt.show()
    
    pred = pred.astype(np.uint8)
    # Overlay the mask on the image
    visualization = overlay_davis(pad_img, pred)  # red
    visualization = Image.fromarray(visualization)
    # visualization = visualization.resize((original_w, original_h))  # 映射回原图尺寸
    #visualization.show()
    visualization.save('./Vis_Show/demo_result.png')
