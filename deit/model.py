import torch
import torch.nn as nn
import timm
import types
from constrained import SWE_Pooling
from fswlib import FSWEmbedding
from swe import SWE_PoolingRun
#Facebooks model is timm.models.vision_transformer VisionTransformer class so you can use all these methods with the model

class ConstrainedDeiT(nn.Module):
    def __init__(self, num_classes, num_ref_points, num_projections, tau_aggregation, parallel, model_name = "deit_tiny_patch16_224", freeze_backbone = True, classification = "constrained_swe", layer_stop = 11, eps = 24):
        super().__init__()

        #DeiT tiny weights
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0) 

        embed_dim = self.backbone.num_features #size of vector before final classification layer (CSW)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.classification = classification

        if self.classification in {"cls", "mean", "GAP"}:
            # CLS uses cls_token, MEAN uses mean over patch tokens
            self.output_dim = embed_dim
        
        elif self.classification == "SWE":
            self.output_dim = num_ref_points + num_projections
            self.pool = SWE_Pooling(
                d_in=embed_dim,
                num_slices=num_projections,
                num_ref_points=num_ref_points,
                alpha_lapsum=10,
                dual_lr=0.001,
                eps=10000,
                tau_aggregation=tau_aggregation,
                parallelized = True)

        # elif self.classification == "SWE":
        #     self.output_dim = num_ref_points + num_projections
        #     self.pool = SWE_PoolingRun(
        #         d_in=embed_dim,
        #         num_projections=num_projections,
        #         num_ref_points=num_ref_points)


        elif self.classification == "constrained_swe":
            self.output_dim = num_ref_points + num_projections
            self.pool = SWE_Pooling(
                d_in=embed_dim,
                num_slices=num_projections,
                num_ref_points=num_ref_points,
                alpha_lapsum=10,
                dual_lr=0.001,
                eps=eps,
                tau_aggregation=tau_aggregation,
                parallelized = True)
        
        elif self.classification == "FSW":
            self.output_dim = num_ref_points + num_projections 
            self.pool = FSWEmbedding(d_in=embed_dim, d_out=self.output_dim, device='cuda')
        
        

        else: #Flattened SWE/CSWE
            # if embedding == "flatten":
            #     self.output_dim = num_ref_points * num_projections 
            # elif embedding == "mapM" or embedding == "meanM":
            #     self.output_dim = num_ref_points
            # elif embedding == "mapL" or embedding == "meanL":
            #     self.output_dim = num_projections

            # # Dual and slack variables
            # self.register_buffer('lambdas', torch.zeros(num_projections))#makes sure its not a parameter (no gd) 
            # self.slacks = nn.Parameter(torch.zeros(num_projections))
            print("hi")

        
        # Final classifier; this is just once at the end so we can make our output embedding of the pooling as high as we want
        self.classifier = nn.Linear(self.output_dim, num_classes)
        self.layer_stop = layer_stop
    
    def train(self, mode = True):  #Keeps backbone frozen
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        # x: B x 3 x H x W
        #This will be the features at certain layer
        #Add normalization at the end of the transformer to be consistent with forward_features (makes sure certain attributes arent giant/too small)
        tokens, cls_token = self.backbone.get_intermediate_layers(x, n=[self.layer_stop], reshape = False, return_prefix_tokens=True, norm = True)[0]
        
        # tokens: B x N x D  CLS: B x 1 x D
        cls_token = cls_token.view(cls_token.size(0), -1)  # [B, D]

        if self.classification == "cls": 
            logits = self.classifier(cls_token) # B x D
            return logits
    
        elif self.classification == "constrained_swe" or self.classification == "SWE": 
            pooled, violations, lambda_prev = self.pool(tokens) # tokens: B x N x D
            logits = self.classifier(pooled)
            return logits, violations, lambda_prev
        
        # elif self.classification == "SWE": 
        #     pooled = self.pool(tokens) # tokens: B x N x D
        #     logits = self.classifier(pooled)
        #     return logits
        
        elif self.classification == "FSW":
            pooled = self.pool(tokens) # tokens: B x N x D
            logits = self.classifier(pooled.reshape(-1, self.output_dim))
            return logits
        
        elif self.classification == "GAP": 
            averaged = tokens.mean(dim=1) #mean of all the tokens in each batch
            logits = self.classifier(averaged)
            return logits