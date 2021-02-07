import torch
state_dict = torch.load('work_dirs/vfnet_RepVGG_5x5_6cls/epoch_90.pth')
state_dict.pop('optimizer')
# state_dict['meta']['epoch'] = 0
# state_dict['meta']['iter'] = 0
torch.save(state_dict, 'work_dirs/vfnet_RepVGG_5x5_6cls/finetune.pth')