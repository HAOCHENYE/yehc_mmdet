import torch
state_dict = torch.load('work_dirs/gflv2_deploy_OSACSP_yefpn_sharedhead/latest.pth')
state_dict.pop('optimizer')
# state_dict['meta']['epoch'] = 0
# state_dict['meta']['iter'] = 0
torch.save(state_dict, 'work_dirs/gflv2_deploy_OSACSP_yefpn_sharedhead/finetune.pth')