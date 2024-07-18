from ff_mod.networks.base_network import AnalogNetwork

from ff_mod.overlay import AppendToEndOverlay

import torch
class Base_A1_LDS(AnalogNetwork):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size, dims, internal_epoch = 2, conv=False) -> None:
        # TODO Refactor the extra pattern size
        super().__init__(
            overlay_function,
            dims = dims,
            loss_function=loss_function,
            internal_epoch=internal_epoch
        )

    def get_latent(self, x, label, depth, reduce = False, unsupervised = False, overlay_factor = 1):
        
        if depth > len(self.layers):
            raise ValueError("Depth should not be greater than the number of layers")

        if not unsupervised:
            h = self.overlay_function(x, label)
        else:
            h = x.clone().detach()
        h = self.adjust_data(h)

        for i, layer in enumerate(self.layers):
            if i == depth:
                return h

            h = layer(h)

        return h
    
    def save_network(self, path):
        torch.save(self.state_dict(), path)
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.save(path + "_overlay_function")
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.load(path + "_overlay_function")
        


class A1_LDS_ANN(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512, mid_size=400) -> None:

        if not isinstance(overlay_function, AppendToEndOverlay):
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784, mid_size, mid_size],
                internal_epoch = 10
            )
        else:
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784 + overlay_function.pattern_size, mid_size, mid_size],
                internal_epoch = 10
            )

class A1_LDS_ANN_Medium(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512) -> None:

        if not isinstance(overlay_function, AppendToEndOverlay):
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784, 800, 800],
                internal_epoch = 10
            )
        else:
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784 + overlay_function.pattern_size, 800, 800],
                internal_epoch = 10
            )


class Cifar_Net_ANN(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512) -> None:

        if not isinstance(overlay_function, AppendToEndOverlay):
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784*3, 800, 800],
                internal_epoch = 10
            )
        else:
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784*3 + overlay_function.pattern_size, 800, 800],
                internal_epoch = 10
            )