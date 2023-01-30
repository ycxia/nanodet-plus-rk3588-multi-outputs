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

Usage: ./build-linux_RK3588.sh

       ./install/rknn_nanodet_demo_Linux/rknn_nanodet_demo nanodet-EfficientNet-Lite2_512_notrs.rknn ../rknn_nanodet_demo/model/bus.jpg
