import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

# 1. Define damage class labels
class_names = ['crack', 'scratch', 'tire flat', 'dent', 'glass shatter', 'lamp broken']

# 2. Load Image Processor & Define Transform
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# 3. Load Model Configuration with num_labels=6
config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=6)
model = ViTForImageClassification(config)
model.load_state_dict(torch.load('car_damage_vit.pth', map_location=torch.device('cpu')))
model.eval()

# 4. Prediction Function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = class_names[predicted_class]

    return predicted_label

# 5. Example Usage
image_path = "C:/summer-project/VICE/data/vaibhavisgreat.png"
prediction = predict_image(image_path)
print(f"Predicted Damage Type: {prediction}")
