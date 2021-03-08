from models import feat3dnet, feat3dnet2, feat3dnet3, feat3dnet4, feat3dnet5


networks_map = {'3DFeatNet': feat3dnet.Feat3dNet,\
		'3DFeatNet2': feat3dnet2.Feat3dNet,
                '3DFeatNet3': feat3dnet3.Feat3dNet, # unsupervised keyDes failed
                '3DFeatNet4': feat3dnet4.Feat3dNet, # atention weight *feat
                '3DFeatNet5': feat3dnet5.Feat3dNet, # concat attentive feat, feat, pool feat
                }


def get_network(name):

    model = networks_map[name]
    return model
