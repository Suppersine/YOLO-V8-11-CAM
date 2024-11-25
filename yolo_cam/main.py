from yolo_cam.grad_cam import GradCAM
from yolo_cam.eigen_cam import EigenCAM

# Example usage
if cam_type == 'gradcam':
    cam = GradCAM(model, target_layers)
elif cam_type == 'eigencam':
    cam = EigenCAM(model, target_layers)
else:
    raise ValueError("Unsupported CAM type!")
