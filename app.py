import torch
from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# load the model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    print(jsonify({'features': features.numpy().tolist()}))

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
