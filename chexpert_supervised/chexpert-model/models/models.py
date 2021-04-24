import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F


from torchvision import models

class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""
    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel must implement forward.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers: List of names for the boundary layers.
            lrs: List of learning rates for each parameter group, from earlier to later layers.

        Returns:
            param_groups: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError('Got {} param groups, but {} learning rates'.format(boundary_layers + 1, len(lrs)))

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i+2]
            param_groups.append({'params': gen_params(start, end), 'lr': lrs[i]})

        return param_groups

    def set_require_grad_for_fine_tuning(model, fine_tune_layers):
        """Set require_grad=False for layers not in fine_tune_layers

        Since we are simply setting model parameters to not require gradient, we do not need to 
        change interaction with optimizer, like the `fine_tuning_parameters` implementation.

        Some design is referenced from Moco lincls
        https://github.com/facebookresearch/moco/blob/master/main_lincls.py

        To keep the implementation generic, you need to know name of layers in the model.
        For example, 
        if 'resnet' in model_name:
            fine_tune = 'module.fc.weight,module.fc.bias'
        elif 'densenet' in model_name:
            fine_tune = 'module.model.classifier.weight,module.model.classifier.bias'
        else:
            raise NotImplementedError

        Args:
            fine_tune_layer: List of layers to fine tune
        
        Returns:
            None
        """

        for name, param in model.named_parameters():
            print(name)
            if name not in fine_tune_layers:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == len(fine_tune_layers)


class CadeneModel(PretrainedModel):
    """Models from Cadene's GitHub page of pretrained networks:
        https://github.com/Cadene/pretrained-models.pytorch
    """
    def __init__(self, model_name, tasks, model_args):
        super(CadeneModel, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty

        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.last_linear.in_features
        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)
        self.fc = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        emb = self.model.features(x)
        # x = F.relu(x, inplace=True)
        x = F.relu(emb, inplace=False)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x, emb
        
# TODO: JBY, WARNING! This thing ain't gonna fly with resnet models, no wonder resnet stuff
# are all Cadene model T_T
class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """
    def __init__(self, model_fn, tasks, model_args):
        super(TorchVisionModel, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty

        self.model = model_fn(pretrained=model_args.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # TODO: JBY
        if 'res' in model_args.model.lower():
            num_ftrs = self.model.fc.in_features
        elif 'mnas' in model_args.model.lower():
            num_ftrs = self.model.classifier._modules['1'].in_features
        else:
            num_ftrs = self.model.classifier.in_features
            
        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)

        # TODO: JBY
        if 'res' in model_args.model.lower():
            self.model.__dict__['classifier'] = nn.Linear(num_ftrs, num_outputs)
        elif 'mnas' in model_args.model.lower():
            self.model.classifier = nn.Linear(num_ftrs, num_outputs)
        else:
            self.model.classifier = nn.Linear(num_ftrs, num_outputs)
        
        self.model_args = model_args

    def forward(self, x):
        if 'mnas' in self.model_args.model.lower():
            emb = self.model.layers(x)
        else:
            emb = self.model.features(x)
        x = F.relu(emb, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)
        # import pdb; pdb.set_trace()
        return x, emb


class DenseNet121(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet121, self).__init__(models.densenet121, tasks, model_args)


class DenseNet161(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet161, self).__init__(models.densenet161, tasks, model_args)


class DenseNet201(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet201, self).__init__(models.densenet201, tasks, model_args)


class ResNet101(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet101, self).__init__(models.resnet101, tasks, model_args)


class ResNet152(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet152, self).__init__(models.resnet152, tasks, model_args)


class Inceptionv3(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(Inceptionv3, self).__init__(models.densenet121, tasks, model_args)


class Inceptionv4(CadeneModel):
    def __init__(self, tasks, model_args):
        super(Inceptionv4, self).__init__('inceptionv4', tasks, model_args)


class ResNet18(CadeneModel):
    def __init__(self, tasks, model_args):
        super(ResNet18, self).__init__('resnet18', tasks, model_args)


class ResNet34(CadeneModel):
    def __init__(self, tasks, model_args):
        super(ResNet34, self).__init__('resnet34', tasks, model_args)

# TODO: JBY
'''
class ResNet50(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet50, self).__init__(models.resnet50, tasks, model_args)
'''
class ResNet50(CadeneModel):
    def __init__(self, tasks, model_args):
        super(ResNet50, self).__init__('resnet50', tasks, model_args)

class ResNeXt101(CadeneModel):
    def __init__(self, tasks, model_args):
        super(ResNeXt101, self).__init__('resnext101_64x4d', tasks, model_args)


class NASNetA(CadeneModel):
    def __init__(self, tasks, model_args):
        super(NASNetA, self).__init__('nasnetalarge', tasks, model_args)

'''
class MNASNet(CadeneModel):
    def __init__(self, tasks, model_args):
        super(MNASNet, self).__init__('nasnetamobile', tasks, model_args)
'''


class MNASNet(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(MNASNet, self).__init__(models.mnasnet1_0, tasks, model_args)


class SENet154(CadeneModel):
    def __init__(self, tasks, model_args):
        super(SENet154, self).__init__('senet154', tasks, model_args)


class SEResNeXt101(CadeneModel):
    def __init__(self, tasks, model_args):
        super(SEResNeXt101, self).__init__('se_resnext101_32x4d', tasks, model_args)
