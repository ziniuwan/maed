import torch.nn as nn

from torchvision.models import resnet50
from lib.utils.utils import determine_output_feature_dim
from lib.models.ktd import KTD
from lib.models.spin import Regressor as Iterative
from lib.models.vision_transformer import vit_custom_resnet50_224_in21k

class MAED(nn.Module):
    def __init__(self, 
        encoder='ste', num_blocks=6, num_heads=12, st_mode='parallel',
        decoder='ktd', hidden_dim=1024, 
        **kwargs):
        super(MAED, self).__init__()

        self._init_encoder(encoder, num_blocks, num_heads, st_mode, **kwargs)
        self._init_decoder(decoder, hidden_dim, **kwargs)


    def _init_decoder(self, decoder, hidden_dim=1024, **kwargs):
        _, feat_dim = determine_output_feature_dim(inp_size=(1, 3, 224, 224), model=self.encoder)
        
        self.decoder_type = decoder
        if decoder.lower() == 'ktd':
            self.decoder = KTD(feat_dim=feat_dim, hidden_dim=hidden_dim, **kwargs)
        elif decoder.lower() == 'iterative':
            self.decoder = Iterative(feat_dim=feat_dim, hidden_dim=hidden_dim, **kwargs)
        else:
            raise NotImplementedError(decoder)
        

    def _init_encoder(self, encoder, num_blocks, num_heads, st_mode, **kwargs):

        self.encoder_type = encoder
        if encoder.lower() == 'cnn':
            self.encoder = resnet50(pretrained=True)
            self.encoder.fc = nn.Identity()
        elif encoder.lower() == 'ste':
            self.encoder = vit_custom_resnet50_224_in21k(num_blocks, num_heads, st_mode, num_classes=-1)
        else:
            raise NotImplementedError(encoder)

    def extract_feature(self, x):

        batch_size, seqlen = x.shape[:2]

        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # (N,T,3,H,W) -> (NT,3,H,W)
        xf = self.encoder(x)
        xf = xf.reshape(batch_size, seqlen, -1)
        return xf

    def forward(self, x, J_regressor=None, **kwargs):
        batch_size, seqlen = x.shape[:2]

        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # (N,T,3,H,W) -> (NT,3,H,W)
        
        xf = self.encoder(x, seqlen=seqlen) if self.encoder_type == 'ste' else self.encoder(x) #(NT, 2048, 7, 7)
        
        output = self.decoder(xf, seqlen=seqlen, J_regressor=J_regressor, **kwargs)

        output['theta']  = output['theta'].reshape(batch_size, seqlen, -1)
        output['verts'] = output['verts'].reshape(batch_size, seqlen, -1, 3)
        output['kp_2d'] = output['kp_2d'].reshape(batch_size, seqlen, -1, 2)
        output['kp_3d'] = output['kp_3d'].reshape(batch_size, seqlen, -1, 3)
        output['rotmat'] = output['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return output