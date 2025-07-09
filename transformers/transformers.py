import numpy as np
import math


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        
        # TODO: Compute attention scores with proper scaling
        scores = None
        
        if mask is not None:
            scores += mask * -1e9
        
        # TODO: Apply softmax to get attention weights
        attention_weights = None
        
        # TODO: Apply attention weights to values
        output = None
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)  
        V = np.matmul(value, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = np.matmul(attention_output, self.W_o)
        
        return output
    
    def softmax(self, x, axis=-1):
        # TODO: Implement numerically stable softmax
        # Hint: subtract max for numerical stability
        pass


class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        self.d_model = d_model
        
        # TODO: Create positional encoding matrix
        # Use sinusoidal functions: sin for even indices, cos for odd indices
        pe = np.zeros((max_seq_length, d_model))
        
        # TODO: Fill in the positional encoding
        # Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        self.pe = pe
    
    def forward(self, x):
        # TODO: Add positional encoding to input
        # Handle batch dimension properly
        pass


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        # TODO: Implement layer normalization
        # Steps:
        # 1. Compute mean and variance across last dimension
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        # 3. Apply learnable scale (gamma) and shift (beta)
        pass


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # TODO: Implement feed forward network
        # Structure: Linear -> ReLU -> Linear
        # FFN(x) = ReLU(xW1 + b1)W2 + b2
        pass
    
    def relu(self, x):
        return np.maximum(0, x)


class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # TODO: Implement encoder layer with residual connections
        # Structure: 
        # 1. attn_output = MultiHeadAttention(x, x, x, mask)
        # 2. x = LayerNorm(x + attn_output)  # First residual connection
        # 3. ff_output = FeedForward(x)
        # 4. x = LayerNorm(x + ff_output)    # Second residual connection
        pass


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # TODO: Implement decoder layer
        # Structure:
        # 1. self_attn_output = SelfAttention(x, x, x, self_attention_mask)
        # 2. x = LayerNorm(x + self_attn_output)
        # 3. cross_attn_output = CrossAttention(x, encoder_output, encoder_output, cross_attention_mask)
        # 4. x = LayerNorm(x + cross_attn_output)
        # 5. ff_output = FeedForward(x)
        # 6. x = LayerNorm(x + ff_output)
        pass


class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.encoder_embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.decoder_embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and decoder layers
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
    
    def embed_tokens(self, token_ids, embedding_matrix):
        # TODO: Convert token indices to embeddings
        # Hint: Use fancy indexing with embedding_matrix[token_ids]
        pass
    
    def create_padding_mask(self, seq, pad_token=0):
        # TODO: Create mask for padded tokens
        # Return mask where padded positions are True (will be masked out)
        pass
    
    def create_look_ahead_mask(self, seq_len):
        # TODO: Create causal mask for decoder self-attention
        # Should be upper triangular matrix to prevent looking at future tokens
        # Return shape: (seq_len, seq_len)
        pass
    
    def forward(self, encoder_input, decoder_input):
        # TODO: Implement full forward pass
        
        # Step 1: Embed tokens and add positional encoding
        # encoder_embedded = ...
        # decoder_embedded = ...
        
        # Step 2: Create masks
        # encoder_padding_mask = ...
        # decoder_padding_mask = ...
        # look_ahead_mask = ...
        
        # Step 3: Pass through encoder layers
        # encoder_output = encoder_embedded
        # for layer in self.encoder_layers:
        #     encoder_output = layer.forward(encoder_output, encoder_padding_mask)
        
        # Step 4: Pass through decoder layers
        # decoder_output = decoder_embedded
        # for layer in self.decoder_layers:
        #     decoder_output = layer.forward(decoder_output, encoder_output, 
        #                                   look_ahead_mask, encoder_padding_mask)
        
        # Step 5: Apply output projection to get logits
        # logits = ...
        
        pass


# Example usage and testing
if __name__ == "__main__":
    # Test individual components first
    print("Testing individual components...")
    
    # Test parameters
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8
    vocab_size = 1000
    
    # TODO: Test each component as you implement it
    # Example:
    # mha = MultiHeadAttention(d_model, num_heads)
    # x = np.random.randn(batch_size, seq_len, d_model)
    # output = mha.forward(x, x, x)
    # print(f"MultiHeadAttention output shape: {output.shape}")
    
    # Test full transformer
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=8,
        num_layers=2,
        d_ff=256,
        max_seq_length=100
    )
    
    # Create dummy input (token indices)
    encoder_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Transformer skeleton created!")
    print("Ready to implement the missing pieces.")


# IMPLEMENTATION ORDER SUGGESTION:
# 1. softmax() - needed by attention
# 2. scaled_dot_product_attention() - core attention mechanism
# 3. PositionalEncoding.forward() - needed for embeddings
# 4. LayerNorm.forward() - needed by all layers
# 5. FeedForward.forward() - needed by encoder/decoder layers
# 6. EncoderLayer.forward() - simpler than decoder
# 7. DecoderLayer.forward() - builds on encoder layer
# 8. Transformer helper methods (embed_tokens, create_masks)
# 9. Transformer.forward() - ties everything together

# TESTING TIPS:
# - Test with very small dimensions first (d_model=4, seq_len=3)
# - Print shapes frequently: print(f"Shape after operation: {x.shape}")
# - Test each component in isolation before combining
# - Use simple inputs like np.ones() to verify basic functionality