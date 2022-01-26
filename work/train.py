import sid

# import ssim
import msssim
import dataloader
import paddle
import paddle.nn as nn
import os

train_path = 'data/train/moire_train_dataset'
# print(train_path)
train_data = dataloader.TrainData(train_path, patch_size=512, file_type='*', hr_dir='gts', lr_dir='images', scale = 1, start=100, end=999999)
train_loader = paddle.io.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
val_path = train_path
val_data = dataloader.ValData(val_path, file_type='*', hr_dir='gts', lr_dir='images', start=0, end=100)
val_loader = paddle.io.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

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

save_path = 'ckpts/sid'
net = sid.UNetSeeInDark(in_channels=3, out_channels=3)
os.makedirs(save_path, exist_ok=True)

load_pretrain=0
learning_rate = 1e-4

if load_pretrain:
    net_path = 'model_best.pdiparams'
    print('load ' + net_path)
    weights = paddle.load(os.path.join(net_path))
    net.load_dict(weights)

optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=net.parameters(), weight_decay=0.0)

mse = nn.MSELoss()
l1 = nn.L1Loss()
ms_ssim_loser = msssim.MS_SSIMLoss(data_range=1.)#.cuda() #####################################
iters = 0
best_score = 0
best_psnr = 0
best_ssim = 0
for epoch in range(2000):
    train_loss = 0
    net.train()
    if epoch > 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    for i, (lr, hr, lrp, hrp) in enumerate(train_loader):
        # print(lr.shape, hr.shape)
        res = net(lr)
        # loss = l1(res*255.0, hr*255.0) #l1(res*255.0, hr*255.0)
        loss = -(0.5 * paddle.log10(65025.0 / mse(res*255.0, hr*255.0)) * 10 /100  + 0.5 * ms_ssim_loser(res, hr).mean())
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_loss += loss.item()
        iters += 1
        # current_lr = optimizer.param_groups[0]['lr']
        current_lr = optimizer.get_lr()
        if iters % 100 ==0:
            print('epoch:', epoch, 'iter:', iters, 'loss:', train_loss/(i+1), 'lr:', current_lr, 'net:', save_path)
    paddle.save(net.state_dict(), '{}/model_latest.pdiparams'.format(save_path))
    if epoch % 1 == 0:
        val_psnr = 0
        val_ms_ssim = 0
        val_score = 0
        net.eval()
        with paddle.no_grad():
            for i, (lr, hr, lrp, hrp) in enumerate(val_loader):
                # lr2 = paddle.transpose(lr, (0,2,3,1)).squeeze().cpu().numpy()
                # lr2 = lr.permute(0,2,3,1).squeeze().cpu().numpy()
                if os.path.basename(save_path) not in ['rcan']:
                    _,_,h,w = lr.shape
                    rh, rw = h, w
                    div = 32
                    pad_h = div - h%div if h%div != 0 else 0
                    pad_w = div - w%div if w%div != 0 else 0
                    m = nn.Pad2D((0,pad_w,0, pad_h), mode='reflect')
                    lr = m(lr)
                    # hr = m(hr)
                # print(lrp, hrp)
                # print(lr.shape, hr.shape)
                res = net(lr)
                # res += rot90(net(rot90(lr,1,[2,3])),3,[2,3])
                # res += rot90(net(rot90(lr,2,[2,3])),2,[2,3])
                # res += rot90(net(rot90(lr,3,[2,3])),1,[2,3])

                
                # res += net(lr.flip([3])).flip([3])
                # res += rot90(net(rot90(lr.flip([3]),1,[2,3])),3,[2,3]).flip([3])
                # res += rot90(net(rot90(lr.flip([3]),2,[2,3])),2,[2,3]).flip([3])
                # res += rot90(net(rot90(lr.flip([3]),3,[2,3])),1,[2,3]).flip([3])
                # res = res / 8

                res = res[:,:,:rh,:rw]
                # res = paddle.where(res > 1, 1, res)
                # res = paddle.where(res < 0, 0, res)
                # print(res)
                loss = mse((res * 255.0).round(), hr*255.0)
                psnr = paddle.log10(65025.0 / loss) * 10
                ms_ssim = ms_ssim_loser(res, hr).mean()
                psnr = psnr.cpu().numpy()
                ms_ssim = ms_ssim.cpu().numpy()
                score = psnr / 100 * 0.5 + ms_ssim * 0.5
                val_psnr += psnr
                val_ms_ssim += ms_ssim
                val_score += score
                # if i % 20 ==0:
                #     print(lrp, hrp)
                #     print('i:', i, 'psnr:', psnr, 'ms_ssim:', ms_ssim, 'score:', score)
                # if i==1:
                #     break
            ave_psnr = val_psnr/(i+1)
            ave_ms_ssim = val_ms_ssim/(i+1)
            ave_score = val_score/(i+1)
            if ave_score > best_score:
                best_score = ave_score
                # print(save_path, ave_psnr, ave_ms_ssim[0], ave_score[0])
                # paddle.save(net.state_dict(), '{}/model_{:.2f}_{:.4f}_{:.4f}.pdiparams'.format(save_path, ave_psnr[0], ave_ms_ssim[0], ave_score[0]))
                paddle.save(net.state_dict(), '{}/model_best.pdiparams'.format(save_path))
            # if ave_psnr > best_psnr:
            #     best_psnr = ave_psnr
            #     paddle.save(net.state_dict(), '{}/model_{:.2f}_{:.4f}_{:.4f}.pdiparams'.format(save_path, ave_psnr[0], ave_ms_ssim[0], ave_score[0]))
            # if ave_ms_ssim > best_ssim:
            #     best_ssim = ave_ms_ssim
            #     paddle.save(net.state_dict(), '{}/model_{:.2f}_{:.4f}_{:.4f}.pdiparams'.format(save_path, ave_psnr[0], ave_ms_ssim[0], ave_score[0]))
            print('epoch:', epoch, 'iter:', iters, 'ave_psnr:', ave_psnr[0], 'ave_ms_ssim:', ave_ms_ssim[0], 'ave_score:', ave_score[0], 'best_score:', best_score, 'best_psnr:', best_psnr, 'best_ssim:', best_ssim)
