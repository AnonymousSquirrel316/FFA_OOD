from ff_mod.networks.spiking_network import SpikingNetwork, SpikingNetworkConv

import torch


from ff_mod.overlay import AppendToEndOverlay


class Base_A1_LDS(SpikingNetwork):

    def __init__(
            self,
            overlay_function,
            loss_function,
            num_classes,
            batch_size,
            dims,
            internal_epoch = 10,
            first_prediction_layer = 0
        ) -> None:
        # TODO Refactor the extra pattern size
        super().__init__(
            overlay_function,
            loss_function,
            dims = dims,
            batch_size=batch_size,
            num_classes=num_classes,
            first_prediction_layer=first_prediction_layer,
            internal_epoch=internal_epoch
        )

    def get_latent(self, x, label, depth, reduce = False, unsupervised = False, overlay_factor = 1):
        
        if depth > len(self.layers):
            raise ValueError("Depth should not be greater than the number of layers")

        if not unsupervised:
            h = self.overlay_function(x, label)
            h[:, 784:] *= overlay_factor
        else:
            h = x.clone().detach()
        h = self.adjust_data(h)

        for i, layer in enumerate(self.layers):
            if i == depth:
                break

            h = layer(h)

        if reduce:
            return h.mean(1)
            
        return h
    
    def save_network(self, path):
        torch.save(self.state_dict(), path)
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.save(path + "_overlay_function")
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.load(path + "_overlay_function")

class A1_LDS(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512, mid_size = 400) -> None:
        # TODO Refactor the extra pattern size
        if not isinstance(overlay_function, AppendToEndOverlay):
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784, mid_size],
                first_prediction_layer=0,
                internal_epoch = 10
            )
        else:
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784 + overlay_function.pattern_size, mid_size],
                first_prediction_layer=0,
                internal_epoch = 10
            )


class A1_LDS_SNN_Medium(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512) -> None:
        # TODO Refactor the extra pattern size
        if not isinstance(overlay_function, AppendToEndOverlay):
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784, 800, 800],
                first_prediction_layer=0,
                internal_epoch = 10
            )
        else:
            super().__init__(
                overlay_function,
                loss_function,
                num_classes,
                batch_size,
                [784 + overlay_function.pattern_size, 800, 800],
                first_prediction_layer=0,
                internal_epoch = 10
            )

class A1_LDS_Large(Base_A1_LDS):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512) -> None:
        # TODO Refactor the extra pattern size
        super().__init__(
            overlay_function,
            loss_function,
            num_classes,
            batch_size,
            [784 + overlay_function.pattern_size, 1200, 1200],
            internal_epoch = 2
        )


class Cifar_Net_SNN(Base_A1_LDS):

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
            


class Base_Conv_FFA(SpikingNetworkConv):

    def __init__(
            self,
            overlay_function,
            loss_function,
            num_classes,
            batch_size,
            dims,
            internal_epoch = 2,
            first_prediction_layer = 0
        ) -> None:
        # TODO Refactor the extra pattern size
        super().__init__(
            overlay_function,
            loss_function,
            dims = dims,
            batch_size=batch_size,
            num_classes=num_classes,
            first_prediction_layer=first_prediction_layer,
            internal_epoch=internal_epoch
        )

    def get_latent(self, x, label, depth, reduce = False, unsupervised = False):
        
        if depth > len(self.layers):
            raise ValueError("Depth should not be greater than the number of layers")

        label_vector = self.overlay_function.label_vectors[label]

        h = x.clone().detach()
        
        h = self.adjust_data(h)

        for i, layer in enumerate(self.layers):
            if i == depth:
                break

            h = layer(h, label_vector=label_vector)

        if reduce:
            return h.mean(1)
            
        return h
    
    def save_network(self, path):
        torch.save(self.state_dict(), path)
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.save(path + "_overlay_function")
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        if isinstance(self.overlay_function, AppendToEndOverlay):
            self.overlay_function.load(path + "_overlay_function")

class A1_Conv(Base_Conv_FFA):

    def __init__(self, overlay_function, loss_function, num_classes, batch_size = 512) -> None:
        # TODO Refactor the extra pattern size
        super().__init__(
            overlay_function,
            loss_function,
            num_classes,
            batch_size,
            [1, 50],
            internal_epoch = 10
        )