from models.EEGModels_tensorflow import MI_EEGNet  # ,ShallowConvNet, DeepConvNet, EEGNet, EEG_TCNet, EEG_inception, EEG_ITNet
# from models.my_model import mymodel1, mymodel2, mymodel3

def Model(modelnum,nb_classes=2,Chans=22,Samples=256):

    # if modelnum == 1:
    #     model = ShallowConvNet(nb_classes, Chans=Chans, Samples=Samples)
    #     modelname = 'ShallowConvNet'
    # elif modelnum == 2:
    #     model = DeepConvNet(nb_classes, Chans=Chans, Samples=Samples)
    #     modelname = 'DeepConvNet'
    # elif modelnum == 3:
    #     model = EEGNet(nb_classes,F1=8,D=2, F2=16, Chans = Chans, Samples=Samples,
    #                    dropoutRate=0.5, kernLength=32)
    #     modelname = 'EEGNet8-2'
    # elif modelnum == 4:
    #     model = EEGNet(nb_classes, F1=4, D=2, F2=8, Chans=Chans, Samples=Samples,
    #                    dropoutRate=0.5, kernLength=32)
    #     modelname = 'EEGNet4-2'
    # elif modelnum == 5:
    #     model = EEG_TCNet(nb_classes, Chans=Chans, Samples=Samples)
    #     modelname = 'EEG_TCNet'
    # elif modelnum == 6:
    #     model = EEG_inception(nb_classes, Chans=Chans, Samples=Samples)
    #     modelname = 'EEG_inception'
    # elif modelnum == 7:
    #     model = EEG_ITNet(nb_classes, Chans=Chans, Samples=Samples)
    #     modelname = 'EEG_ITNet'
    if modelnum == 8:
        model = MI_EEGNet(nb_classes, Chans=Chans, Samples=Samples)
        modelname = 'MI_EEGNet'
    # elif modelnum == 9:
    #     n_ff = [8,8,8,8]
    #     n_sf = [1,1]
    #     model = mymodel1(nb_classes=nb_classes,n_ff=n_ff,n_sf=n_sf, Chans=Chans, Samples=Samples,
    #                dropoutRate = 0.3)
    #     modelname = 'mymodel'

    return model, modelname