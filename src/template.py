def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False
        args.scale = '4'
        args.patch_size = 88
        # args.loss = '0.1*L1+0.8*feature_loss+1e-3*GAN'
        args.loss = '1*L1'
        args.save = 'rcan-4x-Test'
        args.test_only = 'True'
        args.rgb_range = 255
        args.n_colors = 3
        args.save_gt = 'False'
        args.pre_train = '/media/sda/wang/paper/experiment/rcan-4x/model/model_latest.pt'
    if args.template.find('Ours_CON') >= 0:
        args.model = 'Ours_CON'
        args.n_resgroups = 10
        args.n_resblocks = 10
        args.n_feats = 64
        args.chop = False
        args.scale = '3'
        args.patch_size = 66
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'  
        args.save = 'Ours-0427-new_model-latest'
        args.save_gt = ''
        args.test_only = 'True'
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = '/media/sda/wang/paper/experiment/Ours-0427-new_model/model/model_latest.pt'
    if args.template.find('DGSR') >= 0:
        args.model = 'DGSR'
        args.n_resgroups = 2
        args.n_resblocks = 5
        args.n_feats = 64
        args.chop = False
        args.scale = '3'
        args.patch_size = 66
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'  
        args.save = 'DGSR-3x-testwww'
        args.save_gt = 'True'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = ''
    if args.template.find('LWSR') >= 0:
        # print('LWSR')
        args.model = 'LWSR'
        args.n_resgroups = 2
        args.n_resblocks = 5
        args.n_feats = 64
        args.chop = False
        args.scale = '4'
        args.patch_size = 88
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'
        args.save = 'LWSR-4x-light-DW'
        args.save_gt = 'True'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = '/media/sda/wang/paper/experiment/LWSR-4x-light-DW/model/model_best.pt'
    if args.template.find('ESRGAN') >= 0:
        print('esrgan')
        args.model = 'ESRGAN'
        args.n_resgroups = 2
        args.n_resblocks = 5
        args.n_feats = 64
        args.chop = False
        args.scale = '2'
        args.patch_size = 44
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'
        args.save = 'ESRGAN-2x'
        args.save_gt = 'True'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = '/media/sda/wang/paper/experiment/ESRGAN-2x/model/model_latest.pt'
    if args.template.find('DGSR_NOE') >= 0:
        args.model = 'DGSR_NOE'
        args.n_resgroups = 10
        args.n_resblocks = 10
        args.n_feats = 64
        args.chop = False
        args.scale = '3'
        args.patch_size = 66
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'
        args.save = 'DGSR-ablation-without-Encoder'
        args.save_gt = 'True'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = ''
    if args.template.find('DGSR_NOBRANCH2') >= 0:
        args.model = 'DGSR_NOBRANCH2'
        args.n_resgroups = 10
        args.n_resblocks = 10
        args.n_feats = 64
        args.chop = False
        args.scale = '3'
        args.patch_size = 66
        args.loss = '0.1*L1+0.8*feature_loss+1*style_loss+1e-3*GAN'
        # args.loss = '0.1*L1+1*style_loss'
        # args.loss = '1*L1'
        args.save = 'DGSR-ablation-without-branch2'
        args.save_gt = 'True'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 1
        args.pre_train = ''
    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    # if args.template.find('SRGAN') >= 0:
    #     args.model = 'SRGAN'
    #     args.loss = '1.0*feature_loss_original+1e-3*GAN'
    #     args.save = 'test_1216_L1+SRGAN'
    #     args.rgb_range = 1
    #     args.n_colors = 1
    #     args.patch_size = 44
    #     args.scale = '3'
    #     args.patch_size = 66
    #     args.chop = False

    if args.template.find("RDN") >= 0:
        args.model = 'RDN'
        args.save = 'test_rdn_4x-Test'
        args.test_only = 'True'
        args.rgb_range = 255
        args.n_colors = 3
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 88
        args.chop = False
        args.scale = '4'
        args.save_gt = 'False'
        args.pre_train = '/media/sda/wang/paper/experiment/test_rdn_4x/model/model_latest.pt'
    if args.template.find("RFDN") >= 0:
        args.model = 'RFDN'
        args.save = 'iccv_rfdn'
        args.test_only = ''
        args.rgb_range = 255
        args.n_colors = 3
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = False
        args.scale = '4'
        args.pre_train = ''
    # if args.template.find("SRGAN") >= 0:
    #     args.model = 'SRGAN'
    #     args.save = 'test_srgan_4x'
    #     args.test_only = ''
    #     args.rgb_range = 1
    #     args.n_colors = 1
    #     args.n_feats = 64
    #     args.patch_size = 88
    #     args.chop = False
    #     args.scale = '4'
    #     args.pre_train = ''
    if args.template.find("EDSR") >= 0:
        args.model = 'EDSR'
        args.save = 'test_edsr_2x-Test'
        args.test_only = 'True'
        args.rgb_range = 255
        args.n_colors = 3
        args.n_feats = 64
        args.patch_size = 44
        args.chop = False
        args.scale = '2'
        args.save_gt = 'False'
        args.pre_train = '/media/sda/wang/paper/experiment/test_edsr_2x/model/model_latest.pt'