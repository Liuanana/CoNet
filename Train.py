import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.CoNet import CoNet
from utils.dataloader1 import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
#from msloss import msssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def cross_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return (wbce).mean()



def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
 
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5, loss_record6, loss_record7, loss_record8, loss_record9, loss_record10, loss_record1 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, eds, edps = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            eds = Variable(eds).cuda()
            edps = Variable(edps).cuda()
            # ---- rescale ----
      
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #print('gt____________________', gts.size())
                eds = F.upsample(eds, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edps = F.upsample(edps, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, out2h, out3h, out4h, out5v, fuse = model(images)
            #print('yyyyyyyyyyyyyyy',y.size())
            #print('lat2____________________', lateral_map_2.size(), out2h.size())
            lateral_map_edge = torch.mul(lateral_map_2, eds)

            # ---- loss function ----
            loss5 = ms_ssim(lateral_map_5, gts)+structure_loss(lateral_map_5, gts)
            loss4 = ms_ssim(lateral_map_4, gts)+structure_loss(lateral_map_4, gts)
            loss3 = ms_ssim(lateral_map_3, gts)+structure_loss(lateral_map_3, gts)
            loss2 = ms_ssim(lateral_map_2, gts)+structure_loss(lateral_map_2, gts)
            loss1 = ms_ssim(fuse, gts)+structure_loss(fuse, gts)

            #print('lat2____________________', lateral_map_2.size())
            #print('2____________________', out2h.size())
            loss2_2 = structure_loss(out2h, gts)+ms_ssim(out2h, gts)
            loss3_3 = structure_loss(out3h, gts)+ms_ssim(out3h, gts)
            loss4_4 = structure_loss(out4h, gts)+ms_ssim(out4h, gts)
            loss5_5 = structure_loss(out5v, gts)+ms_ssim(out5v, gts)
            #print('gt____________________', gts.size())
            #loss1_1 = structure_loss(pred1, gts)
            #print('output___________', loss2)
            loss6 = cross_loss(lateral_map_edge, edps)
            loss = (loss2 + loss3 + loss4  + 0.6*loss6)/4 + loss2_2/2 + loss3_3/4 + loss4_4/8 + loss5_5/16 + loss5 + loss1 # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
       
            if rate == 1:
                #loss_record1.update(loss1_1.data, opt.batchsize)
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
                loss_record6.update(loss6.data, opt.batchsize)
                loss_record7.update(loss2_2.data, opt.batchsize)
                loss_record8.update(loss3_3.data, opt.batchsize)
                loss_record9.update(loss4_4.data, opt.batchsize)
                loss_record10.update(loss5_5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, lateral-6: {:0.4f}], lateral-7: {:0.4f}], lateral-8: {:0.4f}], lateral-9: {:0.4f}], lateral-10: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), loss_record6.show(), loss_record7.show(), loss_record8.show(), loss_record9.show(), loss_record10.show()))
    save_path = 'snaps/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'CoNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'CoNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=512, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.2, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/Train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='CoNet')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = CoNet().cuda()

    

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    ed_root = '{}/edge/'.format(opt.train_path)
    edp_root = '{}/edgegt/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, ed_root, edp_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
