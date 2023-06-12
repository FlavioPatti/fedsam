import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, Tuple, Optional, Union, Any, List, Sequence
import math
import numpy as np


"""
model:
  classification:
    name: "mobilevit_v2"
    mitv2:
      width_multiplier: 2.0
      attn_norm_layer: "layer_norm_2d"
    activation:
      name: "swish"
  normalization:
    name: "batch_norm"
    momentum: 0.1
  activation:
    name: "swish"
  layer:
    global_pool: "mean"
    conv_init:  "kaiming_normal"
    conv_init_std_dev: 0.02
    linear_init: "trunc_normal"
    linear_init_std_dev: 0.02
"""

Width_multiplier= 2.0
global_pool= "mean"
learn_augmentation_mode=None
activation_name= "swish"
normalization_name= "batch_norm"
normalization_momentum= 0.1
DROPOUT=0
attn_norm_layer="layer_norm_2d"

class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        self.dilation = 1
        output_stride = None
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        self.model_conf_dict = dict()
        self.neural_augmentor = build_neural_augmentor(*args, **kwargs)
        self.gradient_checkpointing = False


    def check_model(self):
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def reset_parameters(self):
        """Initialize model weights"""
        initialize_weights(modules=self.modules())

    def update_classifier(self, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        linear_init_type = "trunc_normal"
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)

            # re-init head
            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer

    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        
        return  layer(x)
    

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Dict[str, Tensor]:
        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy

        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        x = self._forward_layer(self.conv_1, x)  # 112 x112
        x = self._forward_layer(self.layer_1, x)  # 112 x112
        out_dict["out_l1"] = x

        x = self._forward_layer(self.layer_2, x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self._forward_layer(self.layer_3, x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self._forward_layer(self.layer_4, x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self._forward_layer(self.layer_5, x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self._forward_layer(self.conv_1x1_exp, x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        return self.extract_end_points_all(x, use_l5=False)

    def _extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self._forward_layer(self.conv_1, x)
        x = self._forward_layer(self.layer_1, x)
        x = self._forward_layer(self.layer_2, x)
        x = self._forward_layer(self.layer_3, x)

        x = self._forward_layer(self.layer_4, x)
        x = self._forward_layer(self.layer_5, x)
        x = self._forward_layer(self.conv_1x1_exp, x)
        return x

    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self._extract_features(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Any, *args, **kwargs) -> Any:
        if self.neural_augmentor is not None:
            if self.training:
                x_aug = self.neural_augmentor(x)
                prediction = self._forward_classifier(x_aug)  # .detach()
                out_dict = {"augmented_tensor": x_aug, "logits": prediction}
            else:
                out_dict = {
                    "augmented_tensor": None,
                    "logits": self._forward_classifier(x),
                }
            return out_dict
        else:
            x = self._forward_classifier(x, *args, **kwargs)
            return x

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

    @staticmethod
    def _profile_layers(
        layers, input, overall_params, overall_macs, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if layer is None:
                continue
            input, layer_param, layer_macs = module_profile(module=layer, x=input)

            overall_params += layer_param
            overall_macs += layer_macs

            if isinstance(layer, nn.Sequential):
                module_name = "\n+".join([l.__class__.__name__ for l in layer])
            else:
                module_name = layer.__class__.__name__
            print(
                "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                    module_name,
                    "Params",
                    round(layer_param / 1e6, 3),
                    "MACs",
                    round(layer_macs / 1e6, 3),
                )
            )
        return input, overall_params, overall_macs

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        height = 224
        width = 224
        n_labels = 10
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        label_tensor = torch.randint(low=0, high=n_labels, size=(batch_size,)).long()
        return {"samples": img_tensor, "targets": label_tensor}

    def profile_model(
        self, input: Tensor, is_classification: Optional[bool] = True, *args, **kwargs
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], float, float]:
        """
        Helper function to profile a model.

        .. note::
            Model profiling is for reference only and may contain errors as it solely relies on user implementation to
            compute theoretical FLOPs
        """
        overall_params, overall_macs = 0.0, 0.0

        input_fvcore = input.clone()

        if is_classification:
            print("Model statistics for an input of size {}".format(input.size()))
            
            print("{:>35} Summary".format(self.__class__.__name__))

        out_dict = {}
        input, overall_params, overall_macs = self._profile_layers(
            [self.conv_1, self.layer_1],
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l1"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_2,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l2"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_3,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l3"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_4,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l4"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_5,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l5"] = input

        if self.conv_1x1_exp is not None:
            input, overall_params, overall_macs = self._profile_layers(
                self.conv_1x1_exp,
                input=input,
                overall_params=overall_params,
                overall_macs=overall_macs,
            )
            out_dict["out_l5_exp"] = input

        if is_classification:
            classifier_params, classifier_macs = 0.0, 0.0
            if self.classifier is not None:
                input, classifier_params, classifier_macs = module_profile(
                    module=self.classifier, x=input
                )
                print(
                    "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                        "Classifier",
                        "Params",
                        round(classifier_params / 1e6, 3),
                        "MACs",
                        round(classifier_macs / 1e6, 3),
                    )
                )
            overall_params += classifier_params
            overall_macs += classifier_macs

        
            print(
                "{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6)
            )
            overall_params_py = sum([p.numel() for p in self.parameters()])
            print(
                "{:<20} = {:>8.3f} M".format(
                    "Overall parameters (sanity check)", overall_params_py / 1e6
                )
            )

            # Counting Addition and Multiplication as 1 operation
            print(
                "{:<20} = {:>8.3f} M".format(
                    "Overall MACs (theoretical)", overall_macs / 1e6
                )
            )

            # compute flops using FVCore
            try:
                # compute flops using FVCore also
                from fvcore.nn import FlopCountAnalysis

                flop_analyzer = FlopCountAnalysis(self.eval(), input_fvcore)
                flop_analyzer.unsupported_ops_warnings(False)
                flop_analyzer.uncalled_modules_warnings(False)
                flops_fvcore = flop_analyzer.total()

                print(
                    "{:<20} = {:>8.3f} M".format(
                        "Overall MACs (FVCore)**", flops_fvcore / 1e6
                    )
                )
                print(
                    "\n** Theoretical and FVCore MACs may vary as theoretical MACs do not account "
                    "for certain operations which may or may not be accounted in FVCore"
                )
            except Exception:
                pass

            print("Note: Theoretical MACs depends on user-implementation. Be cautious")
    

        return out_dict, overall_params, overall_macs
    
def build_neural_augmentor(*args, **kwargs):
    mode = learn_augmentation_mode

    if mode is None:
        mode = "none"

    mode = mode.lower()
    if mode == "distribution":
        return "this"#DistributionNeuralAugmentor(opts=opts, *args, **kwargs)
    elif mode == "basic":
        return "that"#BasicNeuralAugmentor(opts=opts, *args, **kwargs)
    else:
        return None
def _init_nn_layers(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = None,
) -> None:
    """
    Helper function to initialize neural network module
    """
    init_method = init_method.lower()
    if init_method == "kaiming_normal":
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "kaiming_uniform":
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_normal":
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_uniform":
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "trunc_normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def initialize_conv_layer(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = 0.01,
) -> None:
    """Helper function to initialize convolution layers"""
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_fc_layer(
    module, init_method: Optional[str] = "normal", std_val: Optional[float] = 0.01
) -> None:
    """Helper function to initialize fully-connected layers"""
    if hasattr(module, "layer"):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_weights( modules) -> None:
    """Helper function to initialize differnet layers in a model"""
    # weight initialization
    conv_init_type = "kaiming_normal"
    linear_init_type = "trunc_normal"

    conv_std = 0.02
    linear_std = 0.02
    group_linear_std = 0.01

    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                initialize_conv_layer(
                    module=m, init_method=conv_init_type, std_val=conv_std
                )
          
            elif isinstance(m, (nn.Linear, LinearLayer)):
                initialize_fc_layer(
                    module=m, init_method=linear_init_type, std_val=linear_std
                )
            elif isinstance(m, GroupLinear):
                initialize_fc_layer(
                    module=m, init_method=linear_init_type, std_val=group_linear_std
                )
    else:
        if isinstance(modules, (nn.Conv2d, nn.Conv3d)):
            initialize_conv_layer(
                module=modules, init_method=conv_init_type, std_val=conv_std
            )

        elif isinstance(modules, (nn.Linear, LinearLayer)):
            initialize_fc_layer(
                module=modules, init_method=linear_init_type, std_val=linear_std
            )
        elif isinstance(modules, GroupLinear):
            initialize_fc_layer(
                module=modules, init_method=linear_init_type, std_val=group_linear_std
            )

def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    module_name = kwargs.get("module_name", "")
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [
        {
            "params": with_decay,
            "weight_decay": weight_decay,
            "param_names": with_decay_param_names,
        }
    ]
    if len(without_decay) > 0:
        param_list.append(
            {
                "params": without_decay,
                "weight_decay": 0.0,
                "param_names": without_decay_param_names,
            }
        )
    return param_list

def module_profile(module, x: Tensor, *args, **kwargs) -> Tuple[Tensor, float, float]:
    """
    Helper function to profile a module.

    .. note::
        Module profiling is for reference only and may contain errors as it solely relies on user implementation to
        compute theoretical FLOPs
    """

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                print(e, l)
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)
    return x, n_params, n_macs






class MobileViTv2(BaseEncoder):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, *args, **kwargs) -> None:
        num_classes = 100
        pool_type = global_pool

        mobilevit_config = get_configuration()
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        super().__init__(*args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
             input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
             input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
             input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(

            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
    
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters()

    

    def _make_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                 input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
             input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
         input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = DROPOUT

        block.append(
            MobileViTBlockv2(
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=0,
                attn_dropout=0,
                conv_ksize=3,
                attn_norm_layer=attn_norm_layer,
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel

def get_configuration() -> Dict:

    width_multiplier = Width_multiplier

    ffn_multiplier = (
        2  #bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
    )
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    return config

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(
    min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]
) -> Union[float, int]:
    return max(min_val, min(max_val, value))

class BaseLayer(nn.Module):
    """
    Base class for neural network layers
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


    def forward(self, *args, **kwargs) -> Any:
        pass

    def profile_module(self, *args, **kwargs) -> Tuple[Tensor, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

class LinearLayer(BaseLayer):
    """
    Applies a linear transformation to the input data

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d

    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        channel_first: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()


    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            if not self.training:
                print("Channel-first mode is only supported during inference")
            if x.dim() != 4:
                print("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone()
                    .detach()
                    .reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias,
                )
        else:
            x = F.linear(x, weight=self.weight, bias=self.bias)
        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
            )
        )
        return repr_str

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs


class GroupLinear(BaseLayer):
    """
    Applies a GroupLinear transformation layer, as defined `here <https://arxiv.org/abs/1808.09029>`_,
    `here <https://arxiv.org/abs/1911.12385>`_ and `here <https://arxiv.org/abs/2008.00623>`_

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        n_groups (int): number of groups
        bias (Optional[bool]): use bias or not
        feature_shuffle (Optional[bool]): Shuffle features between groups

    Shape:
        - Input: :math:`(N, *, C_{in})`
        - Output: :math:`(N, *, C_{out})`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: int,
        bias: Optional[bool] = True,
        feature_shuffle: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if in_features % n_groups != 0:
            print(
                "Input dimensions ({}) must be divisible by n_groups ({})".format(
                    in_features, n_groups
                )
            )
        if out_features % n_groups != 0:
            print(
                "Output dimensions ({}) must be divisible by n_groups ({})".format(
                    out_features, n_groups
                )
            )

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.out_features = out_features
        self.in_features = in_features
        self.n_groups = n_groups
        self.feature_shuffle = feature_shuffle

        self.reset_params()

   

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def _forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        # [B, N] -->  [B, g, N/g]
        x = x.reshape(bsz, self.n_groups, -1)

        # [B, g, N/g] --> [g, B, N/g]
        x = x.transpose(0, 1)
        # [g, B, N/g] x [g, N/g, M/g] --> [g, B, M/g]
        x = torch.bmm(x, self.weight)

        if self.bias is not None:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g, B, M/g] --> [B, M/g, g]
            x = x.permute(1, 2, 0)
            # [B, M/g, g] --> [B, g, M/g]
            x = x.reshape(bsz, self.n_groups, -1)
        else:
            # [g, B, M/g] --> [B, g, M/g]
            x = x.transpose(0, 1)

        return x.reshape(bsz, -1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self._forward(x)
            return x
        else:
            in_dims = x.shape[:-1]
            n_elements = x.numel() // self.in_features
            x = x.reshape(n_elements, -1)
            x = self._forward(x)
            x = x.reshape(*in_dims, -1)
            return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, groups={}, bias={}, shuffle={})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.n_groups,
            True if self.bias is not None else False,
            self.feature_shuffle,
        )
        return repr_str

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params = sum([p.numel() for p in self.parameters()])
        macs = params

        out_size = list(input.shape)
        out_size[-1] = self.out_features

        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs
    
class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Defaults to 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. Defaults to 0
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (bool): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``

        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    
class ConvLayer(BaseLayer):
    """
    Applies a 2D convolution over an input

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. When not specified, 
                                               padding is automatically computed based on kernel size 
                                               and dilation rage. Default is ``None``
        groups (Optional[int]): Number of groups in convolution. Default: ``1``
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        if use_norm:
            norm_type = normalization_name
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[0] - 1) / 2) * dilation[0],
                int((kernel_size[1] - 1) / 2) * dilation[1],
            )

        if in_channels % groups != 0:
            print(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            print(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()

        conv_layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = activation_name

        if act_type is not None and use_act:
            neg_slope =  0.1
            inplace =  False
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation


    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        if input.dim() != 4:
            print(
                "Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}".format(
                    input.size()
                )
            )

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, "{}!={}".format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h // stride_h
        out_w = in_w // stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(
            size=(b, self.out_channels, out_h, out_w),
            dtype=input.dtype,
            device=input.device,
        )
        # print(macs)
        return output, params, macs
    
def get_activation_fn(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    return build_activation_layer(
        act_type=act_type,
        num_parameters=num_parameters,
        negative_slope=negative_slope,
        inplace=inplace,
        *args,
        **kwargs
    ) 

def build_activation_layer(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the activation function
    """
    if act_type is None:
        act_type = "none"
    act_type = act_type.lower()
    act_layer = Swish(
            num_parameters=num_parameters,
            inplace=inplace,
            negative_slope=negative_slope,
            *args,
            **kwargs
        )
    
    return act_layer   

class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
    

def get_normalization_layer(
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get normalization layers
    """
    return build_normalization_layer( num_features, norm_type, num_groups)

def build_normalization_layer(
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument
    """
    norm_type = normalization_name
    num_groups = 1
    momentum = normalization_momentum
    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None

    norm_layer = BatchNorm2d(
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    
    return norm_layer

class BatchNorm2d(nn.BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0
    
class GlobalPool(BaseLayer):
    """
    This layers applies global pooling over a 4D or 5D input tensor

    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            print(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        self.pool_type = pool_type
        self.keep_dim = keep_dim


    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)
    
class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        return x, 0.0, 0.0
    
class BaseModule(nn.Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def profile_module(self, input: Any, *args, **kwargs) -> Tuple[Any, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
    
class InvertedResidual(BaseModule):

    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(

                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )
    
class MobileViTBlockv2(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
            groups=in_channels,
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        super(MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.conv_proj = ConvLayer(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize
        self.enable_coreml_compatible_fn = False

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
            
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
             norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_rep), d_model

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
        )
        assert (
            self.patch_h == self.patch_w
        ), "For Coreml, we need patch_h and patch_w are the same"
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm

    def forward_temporal(
        self, x: Tensor, x_prev: Tensor, *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations
        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm, patches

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal data (e.g., videos)
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        return res, params, macs

class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
         embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            get_normalization_layer(
                 norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
             norm_type=norm_layer, num_features=embed_dim
            ),
            ConvLayer(
            
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            Dropout(p=ffn_dropout),
            ConvLayer(
            
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    @staticmethod
    def build_act_layer() -> nn.Module:
        act_type = activation_name
        neg_slope = 0.1
        inplace = False
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, norm_layer={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.norm_name,
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out, p_mha, m_mha = module_profile(module=self.pre_norm_attn, x=input)
        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs
    
class LinearSelfAttention(BaseLayer):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    @staticmethod
    def visualize_context_scores(context_scores):
        # [B, 1, P, N]
        batch_size, channels, num_pixels, num_patches = context_scores.shape

        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
            channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels**0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
        # Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        # [1, 1, H, W] --> [H, W]
        context_map = context_map.squeeze()

        # For ease of visualization, we do min-max normalization
        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)

        try:
            import cv2
            from glob import glob
            import os

            # convert from float to byte
            context_map = (context_map * 255).byte().cpu().numpy()
            context_map = cv2.resize(
                context_map, (80, 80), interpolation=cv2.INTER_NEAREST
            )

            colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
            # Lazy way to dump feature maps in attn_res folder. Make sure that directory is empty and copy
            # context maps before running on different image. Otherwise, attention maps will be overridden.
            res_dir_name = "attn_res"
            if not os.path.isdir(res_dir_name):
                os.makedirs(res_dir_name)
            f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

            files_cmap = glob(
                "{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
            )
            idx = len(files_cmap)
            f_name += str(idx)

            cv2.imwrite("{}.png".format(f_name), colored_context_map)
            return colored_context_map
        except ModuleNotFoundError as mnfe:
            print("Please install OpenCV to visualize context maps")
            return context_map

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)

    def profile_module(self, input) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        if self.out_proj is not None:
            out_p, p, m = module_profile(module=self.out_proj, x=value)
            params += p
            macs += m

        return input, params, macs
    
class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input

    """

    def __init__(
        self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0