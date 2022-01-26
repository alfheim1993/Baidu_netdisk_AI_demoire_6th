import paddle
from network import sid
import numpy as np
import cv2
import os, glob

def np_to_tensor(img):
        assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img)
        img = paddle.to_tensor(img.transpose((2, 0, 1))) / 255.
        return img.unsqueeze(0)  # 255也可以改为256

def rot90(input, k, dims):
    l = len(input.shape)
    new_dims = list(range(l))
    new_dims[dims[0]] = dims[1]
    new_dims[dims[1]] = dims[0]
    flip_dim = min(dims)
    for i in range(k):
        input = paddle.transpose(input, new_dims)
        input = paddle.flip(input, [flip_dim])
    return input

if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
else:
    paddle.set_device('cpu')

test_path = 'data/testB/moire_testB_dataset'
# net_path = 'dm_38.93_0.9928_0.6910.pth'
net_path = 'work/model_best.pdiparams'
net = sid.UNetSeeInDark(in_channels=3, out_channels=3)

print(net_path)
# save_path='./res'
save_path='res'
weights = paddle.load(net_path)
net.load_dict(weights)

net.eval()

# '/test/zhangdy/dataset/baidu/task1/moire_val_dataset'
test_data = sorted(glob.glob(test_path + '/*.*'))
# print(test_data[:10])
os.makedirs(save_path, exist_ok=True)
for i, f in enumerate(test_data):
    print(i,f)
    name = os.path.basename(f).split('.')[0]
    im = cv2.imread(f)
    lr = np_to_tensor(im)
    with paddle.no_grad():
        _,_,h,w = lr.shape
        rh, rw = h, w
        div = 32
        if h%div != 0:
            pad_h = div - h%div
        else:
            pad_h = 0
        if w%div != 0:
            pad_w = div - w%div
        else:
            pad_w = 0
        m = paddle.nn.Pad2D((0,pad_w,0, pad_h), mode='reflect')
        lr = m(lr)#.cuda()
        res = net(lr)
        # 数据增强
        res += rot90(net(rot90(lr,1,[2,3])),3,[2,3])
        res += rot90(net(rot90(lr,2,[2,3])),2,[2,3])
        res += rot90(net(rot90(lr,3,[2,3])),1,[2,3])

        res += net(lr.flip([3])).flip([3])
        res += rot90(net(rot90(lr.flip([3]),1,[2,3])),3,[2,3]).flip([3])
        res += rot90(net(rot90(lr.flip([3]),2,[2,3])),2,[2,3]).flip([3])
        res += rot90(net(rot90(lr.flip([3]),3,[2,3])),1,[2,3]).flip([3])
        res = res / 8

        res = res[:,:,:rh,:rw]
        res = res.cpu().numpy()
        res = np.clip(res * 255, 0, 255)
        res = np.transpose(res, (0, 2, 3, 1))
        res = res[0, :, :, ::-1]
        cv2.imwrite(save_path + '/' + name + '.jpg', res)
