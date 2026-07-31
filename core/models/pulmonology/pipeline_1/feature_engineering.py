import torch

# ==============================================================================
# FEATURE ENGINEERING EXPLAINED
# ==============================================================================
# In Deep Learning (like our DenseNet121 model), "Feature Engineering" is mostly 
# handled automatically by the Convolutional Neural Network (CNN). 
# 
# Traditional Machine Learning (Random Forests, SVM) required humans to manually 
# extract features from an image. 
#
# Below are examples of how we *could* extract features manually if we wanted 
# to build a Hybrid Model (e.g., passing deep features into a Random Forest).

def extract_statistical_features(image_tensor):
    """
    Extracts simple statistical pixel features from an X-ray.
    This is typical for classical Machine Learning models.
    """
    features = {
        'mean_intensity': image_tensor.mean().item(),
        'std_intensity': image_tensor.std().item(),
        'max_pixel': image_tensor.max().item(),
        'min_pixel': image_tensor.min().item()
    }
    return features


@torch.no_grad()
def extract_deep_embeddings(model, image_tensor, device='cuda'):
    """
    Extracts high-level "Deep Features" (Embeddings) from our trained Neural Network.
    
    Instead of getting the final classification (Normal vs COVID), we strip off the 
    last classification layer and extract the raw vector of numbers that the model 
    uses to represent the image conceptually.
    
    This is extremely useful if you want to use the CNN as a "Feature Extractor" 
    and pass these embeddings into a completely different system (like a Vector Database 
    for image retrieval).
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # We can get features from DenseNet before the final classifier
    # model.features outputs the deep CNN maps
    features = model.features(image_tensor) 
    
    # Apply global average pooling to flatten the maps into a 1D vector (Embedding)
    import torch.nn.functional as F
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    embedding = torch.flatten(out, 1)
    
    return embedding
