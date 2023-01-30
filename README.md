# nanodet-plus-rk3588-multi-outputs
Deploy nanodet model which has multi outputs different from original one output model

Nanodet-plus author concat multi outputs into one output, but resize(transpose) layer is time consuming for rk3588.

So, inder to decrease NPU inference time, I deleted concat and transpose layer.

For nanodet-plus head model, when convert pytorch(.pth) model to torchscript(.pt) or .onnx model, you should change some code firstly.

http://github.xiaoc.cn/RangiLyu/nanodet/blob/main/nanodet/model/head/nanodet_plus_head.py#L146

comment the following two lines, and add outputs.append(output).

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output)  # add this line
        ######comment following two lines########
        #    outputs.append(output.flatten(start_dim=2))
        #outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
        
For nanodet-EfficientNet-Lite2_512, the head is nanodet. you should change nanodet_head.py.http://github.xiaoc.cn/RangiLyu/nanodet/blob/0036f941ba34be34db31486a83a18be9da91e293/nanodet/model/head/nanodet_head.py#L156

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)
                output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output)  #add this line
            ######comment following two lines#######
            #outputs.append(output.flatten(start_dim=2))
        #outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
        
After changing,  your converted .pt model has multi outputs.

nanodet-EfficientNet-Lite2_512 has three outputs, each output dims is (1,124,64,64), (1,124,32,32), (1,124,16,16). 
the input dims is(1,3,512,512).

Usage: 

       ./build-linux_RK3588.sh

       ./install/rknn_nanodet_demo_Linux/rknn_nanodet_demo nanodet-EfficientNet-Lite2_512_notrs.rknn bus.jpg
