from yolo_cam.base_cam import BaseCAM

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, task='od', reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layers, task, reshape_transform, uses_gradients=True)

    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        # Grad-CAM weight computation
        return grads.mean(axis=(2, 3))
