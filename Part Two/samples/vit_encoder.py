import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PatchEmbedding(layers.Layer):
    """Convert image into patches and embed them."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    """Transformer block with attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_vit_classifier(
    image_size: int = 32,
    patch_size: int = 4,
    num_layers: int = 8,
    num_classes: int = 100,
    d_model: int = 128,
    num_heads: int = 8,
    ff_dim: int = 256,
    dropout: float = 0.1,
) -> keras.Model:
    """Create Vision Transformer for classification.
    
    Args:
        image_size: Size of input images (32 for CIFAR-100)
        patch_size: Size of each patch
        num_layers: Number of transformer blocks
        num_classes: Number of output classes (100 for CIFAR-100)
        d_model: Dimension of the model
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward network
        dropout: Dropout rate
    
    Returns:
        Keras model
    """
    num_patches = (image_size // patch_size) ** 2
    patch_dim = 3 * patch_size ** 2
    
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    # Augmentation layer (only applied during training)
    augmented = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")(inputs, training=True)
    
    # Create patches
    patches = layers.Reshape((num_patches, patch_dim))(augmented)
    
    # Encode patches
    encoded_patches = PatchEmbedding(num_patches, d_model)(patches)
    
    # Create multiple layers of the Transformer block
    for _ in range(num_layers):
        encoded_patches = TransformerBlock(
            d_model, num_heads, ff_dim, dropout
        )(encoded_patches, training=True)
    
    # Create a [batch_size, projection_dim] tensor
    representation = layers.GlobalAveragePooling1D()(encoded_patches)
    representation = layers.Dropout(dropout)(representation)
    
    # Classification head
    features = layers.Dense(ff_dim, activation='gelu')(representation)
    features = layers.Dropout(dropout)(features)
    outputs = layers.Dense(num_classes)(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_cifar100_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-100 dataset."""
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # Convert to float32 and normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Convert labels to float32
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_learning_rate_scheduler(warmup_epochs: int = 5, total_epochs: int = 50) -> keras.callbacks.Callback:
    """Create a learning rate scheduler with warmup."""
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        else:
            return lr * tf.math.exp(-0.01)
    
    return keras.callbacks.LearningRateScheduler(scheduler)

def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, x_test, y_test, class_names, num_examples=10):
    """Visualize model predictions on test data."""
    # Get random indices
    indices = np.random.choice(len(x_test), num_examples, replace=False)
    
    # Make predictions
    predictions = model.predict(x_test[indices])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[indices].flatten().astype(int)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x_test[idx])
        axes[i].axis('off')
        
        true_label = class_names[true_classes[i]]
        pred_label = class_names[predicted_classes[i]]
        confidence = np.max(predictions[i]) * 100
        
        color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)', 
                         fontsize=10, color=color)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load CIFAR-100 data
    (x_train, y_train), (x_test, y_test) = load_cifar100_data()
    
    # CIFAR-100 class names (shortened for display)
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    # Create Vision Transformer model
    model = create_vit_classifier(
        image_size=32,
        patch_size=4,
        num_layers=6,  # Reduced for faster training
        num_classes=100,
        d_model=128,
        num_heads=8,
        ff_dim=256,
        dropout=0.2,
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    lr_scheduler = create_learning_rate_scheduler(warmup_epochs=5, total_epochs=30)
    
    # Train model
    batch_size = 128
    epochs = 30  # Adjust based on your computational resources
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize predictions
    visualize_predictions(model, x_test, y_test, class_names)
    
    # Save model
    model.save('vit_cifar100_model.h5')
    print("\nModel saved as 'vit_cifar100_model.h5'")
