import torch
from yolo_cam.base_cam import BaseCAM
from yolo_cam.utils.svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, task: str = 'od', #use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       task,
                                       #use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, task: str = 'od', #use_cuda=False,
                 reshape_transform=None):
        super(GradCAM, self).__init__(model,
                                    target_layers,
                                    task,
                                    reshape_transform,
                                    #use_cuda,
                                    uses_gradients=True  # Grad-CAM requires gradients
        )

    def get_cam_weights(self, input_tensor, target_layers, targets, activations, grads):
        """
        Compute weights for Grad-CAM:
        - Averaging gradients across the spatial dimensions.
        """
        # grads: Gradients of the activations w.r.t. the target
        return torch.mean(grads, dim=(2, 3))  # BxC: Batch and Channels
