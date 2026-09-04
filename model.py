import torch
import torch.nn as nn
import math


# NORMAL EMBEDDINGS
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)   # using PYTORCH's inbuilt library for normal embeedings
        
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)  # multiplying by sq-rt as it is specified in oiginal research paper
    
# POSITIONAL ENCODING 
class PositionalEncoding(nn.Module):
    
    # max-length of sentence = seq_len
    def __init__(self, d_model: int, seq_len: int, dropout: float) :
        super().__init__()
        self.d_model =d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create a matrix of shape seq_len*d_model
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)    # numerator of positional encoding formula of original reserch paper
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) # denominator of positional encoading formula of original reserch paper(used slightly different approach(log based) for stability, results remain same)
        # applying SIN function to even terms and COS function to odd terms
        pe[:, 0::2] = torch.sin(position * div_term) # pe[:,0::2] -> (: = for all elements of pe) ; (0::2 = staring from 0 with step of 2, i.e.. 0,2,4,6,8...)
        pe[:, 1::2] = torch.cos(position* div_term)  #  pe[:,1::2] -> (: = for all elements of pe) ; (1::2 = staring from 1 with step of 2, i.e.. 1,3,5,7,9...)
        
        pe = pe.unsqueeze(0) # adding a dimension for batch, now dimension is (1,seq_len, d_model)
        
        # Adding to buffer, BUFFER = when we want to save something but not as learned parammeter
        self.register_buffer("pe",pe)
        
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #adding positional encoading to real embeddings, and making them fixed(telling model that no learnable parameter)
        return self.dropout(x)
    
    
# NORMALIZATION
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))     # multiplier 
        self.bias = nn.Parameter(torch.zeros(1))     # added
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim = True)
        return self.alpha * x-mean/std + self.eps +self.bias
    
# FEED-FORWARD LAYER
class FeedForwardBlock(nn.Module):
    
    def __init__(self,d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)  # 512 -> 2048 (W1,B1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # 2048 -> 512 (W2,B2)
        
    def forward(self,x):
        # (Batch_size, seq_len, d_model)-->(Batch_size, seq_len, d_ff)-->(Batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# MULTI-HEAD ATTENTION
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # spliting embeddings for h different heads so d_model % h, each head will get something from embedding of each word-
        # so we are splitting them in a way that each head will get full sentence but not full embeeding of each word
        assert d_model%h == 0 , "d-model is not divisible by h" # d_model should be divisible by h
        
        self.d_k = d_model//h
        # defining weight matrices which will give key,query and value embeddings
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        self.w_o = nn.Linear(d_model,d_model) # weight matrix of output
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        #apply masking when given 
        if mask is not None:
            attention_score.masked_fill_(mask==0, -1e9)
        
        attention_score = attention_score.softmax(dim = -1)
        
        if dropout is not None:
            attention_score = dropout(attention_score)
            
        return (attention_score @ value), attention_score
        
    def forward(self,q,k,v, mask):
        query = self.w_q(q)  # multiplying original matrix q with weights to get query matrix
        key = self.w_k(k)    # multiplying original matrix k with weights to get key matrix
        value = self.w_v(v)  # multiplying original matrix v with weights to get value matrix
        
        # (batch,seq_len,d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_score = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        # (batch, h,seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x= x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        
        return self.w_o(x)
    
    
# RESIDUEL CONNECTION
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ENCOADER BLOCK    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask,):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    
    def __init__(self,layer: nn.ModuleList):
        super().__init__()
        self.layers = layer
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x,mask)
        return self.norm(x)
        
        
# DECODER BLOCK
class DecoaderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, target_mask))
        x = self.residual_connection[0](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output, src_mask))
        x = self.residual_connection[0](x,self.feed_forward_block)
        return x
    
# DECODER
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,encoader_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoader_output, src_mask, tgt_mask)
            
        return self.norm(x)
    
    
# PROJECTION/ LINEAR LAYER
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        return torch.log_softmax(self.proj(x), dim = -1)


# TRANSFORMER
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, srcc_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = srcc_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len : int, tgt_seq_len: int, d_model: int=512, N:int =6, h: int =8, dropout: float= 0.1, d_ff = 2048)-> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) 
    tgt_pos  = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # ENCODER
    encoder_blocks =[]
    for _ in range(N):
        encoader_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        
        encoader_block = EncoderBlock(encoader_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoader_block)
    
    # DECODER
    decoder_blocks =[]
    for _ in range(N):
        decoader_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoader_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        decoder_block = DecoaderBlock(decoader_self_attention_block,decoader_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
        
    # CREATING ENCODER
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create TRANSFORMER
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialization of parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer