from ultralytics import YOLO


def export_model(model_path: str, output_path: str):
    """
    Exports a YOLO model to ONNX format.

    Args:
        model_path (str): Path to the YOLO model file.
        output_path (str): Path where the ONNX model will be saved.
    """
    try:
        # Load the YOLO model
        model = YOLO(model_path)

        # Export the model to ONNX format
        model.export(format='onnx')

        print(f"Model exported successfully to {output_path}")

    except Exception as e:
        print(f"Error exporting model: {e}")


if __name__ == "__main__":
    # Example usage
    model_path = "models_pose/yolo11x-pose.pt"  # Path to your YOLO model
    # Desired output path for the ONNX model
    output_path = "models_pose/yolo11x-pose.onnx"

    export_model(model_path, output_path)
